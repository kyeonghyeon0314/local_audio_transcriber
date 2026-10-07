"""Gemma 4 멀티모달 모델 로딩과 추론.

원본 오디오 파형과 PDF 문맥(텍스트/페이지 이미지)을 하나의 대화 메시지로 묶어
모델에 직접 넣는다. 별도 STT 단계는 없다.
"""

import logging

log = logging.getLogger(__name__)

# 4bit/8bit 양자화에서 제외할 모듈. 오디오/비전 인코더는 작고 품질에 민감하므로 원래 정밀도로 둔다.
SKIP_QUANT_MODULES = ["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"]


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

        kwargs = {"dtype": torch.bfloat16}
        if use_cuda:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {0: str(cfg.get("max_gpu_memory", "7GiB")), "cpu": "64GiB"}
            quant = self._quant_config(str(cfg.get("quantization", "4bit")).lower())
            if quant is not None:
                kwargs["quantization_config"] = quant
        else:
            kwargs["device_map"] = {"": "cpu"}

        log.info("모델 로딩 중: %s (처음 실행 시 다운로드로 오래 걸릴 수 있습니다)", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = _load_model_class().from_pretrained(model_id, **kwargs)
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
        common = dict(llm_int8_skip_modules=SKIP_QUANT_MODULES, llm_int8_enable_fp32_cpu_offload=True)
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
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
