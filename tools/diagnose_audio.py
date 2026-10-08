"""오디오가 실제로 모델에 전달되는지 확인하는 진단 스크립트.

input 폴더의 첫 번째 녹음에서 앞 25초를 잘라
  1) 채팅 템플릿이 오디오 자리표시 토큰을 넣는지
  2) 오디오 특징(input_features)과 오디오 토큰 수가 맞는지
  3) 가장 단순한 프롬프트로 전사가 되는지
를 출력하고, 같은 내용을 output/logs/diagnose.txt 에 저장한다.
"""

import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from transcriber.audio import SAMPLE_RATE, load_audio  # noqa: E402
from transcriber.config import load_config  # noqa: E402
from transcriber.pipeline import find_inputs  # noqa: E402

LOG = ROOT / "output" / "logs" / "diagnose.txt"
_lines: list[str] = []


def out(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    _lines.append(text)


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    audios, _ = find_inputs([ROOT / "input"])
    if not audios:
        out("input 폴더에 녹음 파일이 없습니다.")
        return
    audio = load_audio(audios[0])
    clip = audio[: 25 * SAMPLE_RATE].astype(np.float32)
    wav_path = LOG.parent / "diagnose_clip.wav"
    save_wav(wav_path, clip)
    out("파일:", audios[0].name)
    out(f"오디오: 전체 {len(audio) / SAMPLE_RATE:.1f}s, 진단 구간 {len(clip) / SAMPLE_RATE:.1f}s, "
        f"dtype={clip.dtype}, min={clip.min():.4f}, max={clip.max():.4f}, rms={np.sqrt(np.mean(clip**2)):.4f}")
    out("진단 구간 WAV 저장 (직접 들어 보세요):", wav_path)

    from transcriber.model import GemmaTranscriber

    model = GemmaTranscriber(cfg)
    proc = model.processor
    tok = proc.tokenizer
    out("audio_token:", repr(getattr(proc, "audio_token", None)), "id:", getattr(proc, "audio_token_id", None),
        "| boa:", repr(getattr(proc, "boa_token", None)), "eoa:", repr(getattr(proc, "eoa_token", None)))

    template = proc.chat_template or getattr(tok, "chat_template", None) or ""
    template = template if isinstance(template, str) else str(template)
    out("\n--- 채팅 템플릿 중 audio 관련 줄 ---")
    for line in template.splitlines():
        if "audio" in line.lower():
            out("   ", line.strip()[:200])

    variants = {
        "A. 오디오 먼저 + 짧은 지시": [
            {"type": "audio", "audio": clip},
            {"type": "text", "text": "Transcribe this audio verbatim in the language spoken."},
        ],
        "B. 짧은 지시 먼저 + 오디오": [
            {"type": "text", "text": "Transcribe this audio verbatim in the language spoken."},
            {"type": "audio", "audio": clip},
        ],
    }
    for name, content in variants.items():
        messages = [{"role": "user", "content": content}]
        rendered = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        shown = rendered
        if proc.audio_token:
            shown = shown.replace(proc.audio_token * 2, "")  # 반복 토큰 축약
        out(f"\n=== {name} ===")
        out("템플릿 렌더링 결과(앞 600자):", repr(shown[:600]))
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = inputs["input_ids"][0]
        n_audio_tok = int((ids == proc.audio_token_id).sum()) if proc.audio_token_id is not None else -1
        out("입력 키:", list(inputs.keys()))
        out("총 토큰:", len(ids), "| 오디오 토큰 수:", n_audio_tok)
        if "input_features" in inputs:
            f = inputs["input_features"]
            out("input_features:", tuple(f.shape), f.dtype, f"mean={f.float().mean():.3f} std={f.float().std():.3f}")
        if "input_features_mask" in inputs:
            out("input_features_mask 합:", int(inputs["input_features_mask"].sum()))
        text = model.generate(content)
        out("모델 출력:", repr(text[:500]))

    LOG.write_text("\n".join(_lines), encoding="utf-8")
    print("\n진단 결과 저장:", LOG)


if __name__ == "__main__":
    main()
