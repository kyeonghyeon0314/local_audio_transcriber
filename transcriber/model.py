"""Gemma 4 멀티모달 모델 로딩과 추론.

원본 오디오 파형과 PDF 문맥(텍스트/페이지 이미지)을 하나의 대화 메시지로 묶어
모델에 직접 넣는다. 별도 STT 단계는 없다.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# 4bit/8bit 양자화에서 제외할 모듈. 오디오/비전 인코더는 작고 품질에 민감하므로 원래 정밀도로 둔다.
SKIP_QUANT_MODULES = ["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"]

# Gemma 4 E2B/E4B의 레이어별 임베딩(Per-Layer Embedding) 테이블.
# 수십억 파라미터짜리 nn.Embedding이라 4bit 양자화 대상이 아니고 bf16으로 수 GB를 차지한다.
# 토큰 조회만 하므로 CPU RAM에 두어도 속도 손실이 거의 없어, GPU에는 나머지만 올린다.
PLE_SUFFIX = "embed_tokens_per_layer"


def _load_model_class():
    try:
        from transformers import AutoModelForMultimodalLM as cls
    except ImportError:
        from transformers import AutoModelForImageTextToText as cls
    return cls


class GemmaTranscriber:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        model_id = cfg["model"]
        device = str(cfg.get("device", "auto")).lower()
        use_cuda = torch.cuda.is_available() and device in ("auto", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config.yaml에서 device: cuda로 설정했지만 CUDA GPU를 사용할 수 없습니다.")

        if use_cuda:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            log.info("GPU: %s (%.1f GB)", name, total)
        else:
            log.warning("GPU를 사용하지 않습니다. CPU 추론은 매우 느립니다.")

        model_cls = _load_model_class()
        self.input_device = torch.device("cuda:0" if use_cuda else "cpu")
        kwargs = {"dtype": torch.bfloat16}
        cpu_modules: list[str] = []
        quant = self._quant_config(str(cfg.get("quantization", "4bit")).lower()) if use_cuda else None
        if quant is not None:
            # bitsandbytes 양자화 레이어는 CPU offload를 지원하지 않으므로 device_map을 직접 정한다:
            # PLE 테이블만 CPU, 나머지는 모두 GPU.
            kwargs["quantization_config"] = quant
            kwargs["device_map"], cpu_modules = _device_map_with_cpu_ple(model_cls, model_id, gpu=0)
        elif use_cuda:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {0: str(cfg.get("max_gpu_memory", "7GiB")), "cpu": "64GiB"}
        else:
            kwargs["device_map"] = {"": "cpu"}

        log.info("모델 로딩 중: %s (처음 실행 시 다운로드로 오래 걸릴 수 있습니다)", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(model_id, **kwargs)
        for name in cpu_modules:
            _run_embedding_on_cpu(self.model, name, model_id, self.input_device)
        self.model.eval()
        if use_cuda:
            log.info("모델 로딩 완료. GPU 메모리 사용량: %.1f GB", torch.cuda.memory_allocated(0) / 2**30)
        else:
            log.info("모델 로딩 완료.")
        self.max_new_tokens = int(cfg["transcription"].get("max_new_tokens", 768))

    def _quant_config(self, quantization: str):
        if quantization in ("none", "", "false", "bf16"):
            return None
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception as e:
            log.warning("bitsandbytes를 불러올 수 없어 양자화 없이 로딩합니다 (부족한 메모리는 CPU로 offload): %s", e)
            return None
        common = dict(llm_int8_skip_modules=SKIP_QUANT_MODULES + [PLE_SUFFIX], llm_int8_enable_fp32_cpu_offload=True)
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, **common)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            **common,
        )

    def generate(self, content: list[dict]) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.input_device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


def _device_map_with_cpu_ple(model_cls, model_id: str, gpu) -> tuple[dict, list[str]]:
    """PLE 테이블만 'cpu', 나머지 모듈은 모두 GPU에 두는 device_map을 만든다."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        empty = model_cls.from_config(config)
    cpu_names = [n for n, _ in empty.named_modules() if n.endswith(PLE_SUFFIX)]
    if not cpu_names:
        return {"": gpu}, []

    device_map: dict = {}

    def walk(module, prefix: str) -> None:
        for pname, _ in module.named_parameters(recurse=False):
            device_map[prefix + pname] = gpu
        for name, child in module.named_children():
            full = prefix + name
            if full in cpu_names:
                device_map[full] = "cpu"
            elif any(c.startswith(full + ".") for c in cpu_names):
                walk(child, full + ".")
            else:
                device_map[full] = gpu

    walk(empty, "")
    return device_map, cpu_names


def _run_embedding_on_cpu(model, name: str, model_id: str, out_device) -> None:
    """임베딩 모듈을 CPU에서 조회하고 결과만 GPU로 보내도록 바꾼다.

    accelerate는 'cpu'로 지정된 모듈을 실행할 때마다 가중치 전체를 GPU로 복사하므로
    (수 GB라 매우 느림) 그 hook을 제거하고 가중치를 CPU에 고정한다.
    """
    import torch
    import torch.nn.functional as F
    from accelerate.hooks import remove_hook_from_module

    module = model.get_submodule(name)
    if hasattr(module, "_hf_hook"):
        remove_hook_from_module(module)
    weight = module.weight
    if weight.device.type != "cpu":
        weight = _load_checkpoint_tensor(model_id, f"{name}.weight")
    module.weight = torch.nn.Parameter(weight.to("cpu", torch.bfloat16), requires_grad=False)
    scale = float(getattr(module, "scalar_embed_scale", 1.0))

    def forward(input_ids):
        out = F.embedding(input_ids.to("cpu"), module.weight) * scale
        return out.to(out_device)

    module.forward = forward
    log.info("%s 를 CPU RAM에 배치했습니다 (%.1f GB).", name, module.weight.numel() * 2 / 2**30)


def _load_checkpoint_tensor(model_id: str, param_name: str):
    """safetensors 체크포인트에서 텐서 하나를 직접 읽는다 (이름 접두사가 달라도 끝부분으로 찾음)."""
    from safetensors import safe_open
    from transformers.utils import cached_file

    index_file = cached_file(model_id, "model.safetensors.index.json", _raise_exceptions_for_missing_entries=False)
    if index_file:
        weight_map = json.loads(Path(index_file).read_text(encoding="utf-8"))["weight_map"]
    else:
        weight_map = None
    keys = list(weight_map) if weight_map else None
    if keys is None:
        single = cached_file(model_id, "model.safetensors")
        with safe_open(single, "pt", device="cpu") as f:
            keys = list(f.keys())
    tail = param_name.rsplit(".", 2)[-2] + ".weight"  # 예: embed_tokens_per_layer.weight
    key = param_name if param_name in keys else next((k for k in keys if k.endswith(tail)), None)
    if key is None:
        raise RuntimeError(f"체크포인트에서 {param_name} 가중치를 찾지 못했습니다.")
    file = cached_file(model_id, weight_map[key]) if weight_map else cached_file(model_id, "model.safetensors")
    with safe_open(file, "pt", device="cpu") as f:
        return f.get_tensor(key)
