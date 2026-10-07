"""config.yaml 로딩. 파일이 없거나 일부 키가 빠져도 기본값으로 동작한다."""

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "model": "google/gemma-4-E4B-it",
    "quantization": "4bit",
    "device": "auto",
    "max_gpu_memory": "7GiB",
    "output": {"format": "txt", "timestamp": True, "open_folder": True},
    "audio": {"chunk_length": 25, "overlap": 5, "skip_silence": True},
    "pdf": {"mode": "auto", "max_text_chars": 12000, "max_images": 8, "image_dpi": 100},
    "transcription": {
        "preserve_language": True,
        "speaker_labels": True,
        "acoustic_events": True,
        "context_lines": 8,
        "max_new_tokens": 768,
    },
}

# Gemma 4 오디오 인코더는 한 번에 최대 30초(750 토큰 x 40ms)까지 받는다.
MAX_CHUNK_SECONDS = 30


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict:
    user_cfg = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user_cfg)

    audio = cfg["audio"]
    audio["chunk_length"] = min(float(audio["chunk_length"]), MAX_CHUNK_SECONDS)
    audio["overlap"] = max(0.0, float(audio["overlap"]))
    if audio["overlap"] >= audio["chunk_length"]:
        raise ValueError("config.yaml: audio.overlap은 audio.chunk_length보다 작아야 합니다.")
    return cfg
