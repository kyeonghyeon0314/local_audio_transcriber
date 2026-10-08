"""녹음 파일 하나를 전사하고 결과 TXT를 만든다.

기본(diarization.enabled): 화자 구분 보조 모델로 화자 턴을 나눈 뒤, 연속된 턴들을 번호 붙인
원본 오디오 여러 개로 묶어 Gemma에 넣는다. 끄면 25초 구간(5초 겹침)을 Gemma만으로 전사한다.
"""

import logging
import time
from pathlib import Path

from .audio import AUDIO_EXTENSIONS, SAMPLE_RATE, load_audio, normalize, split_chunks
from .diarize import diarize, make_windows
from .pdf_context import PdfContext, build_pdf_context
from .transcript import (
    Turn,
    append_turns,
    context_tail,
    parse_numbered,
    parse_output,
    remove_overlap,
    render,
    speakers_seen,
)

log = logging.getLogger(__name__)

# 화자 턴 오디오 앞뒤로 붙이는 여유(초). 구간 경계에서 첫/끝 음절이 잘리지 않게 한다.
CLIP_PAD = 0.15


def find_inputs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(x for x in p.iterdir() if x.is_file()))
        elif p.is_file():
            files.append(p)
    audios = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    pdfs = [f for f in files if f.suffix.lower() == ".pdf"]
    return audios, pdfs


def pick_pdfs(audio: Path, pdfs: list[Path]) -> list[Path]:
    """녹음과 파일명이 같은 PDF가 있으면 그것만, 없으면 전체 PDF를 사용한다."""
    same = [p for p in pdfs if p.stem == audio.stem]
    return same or pdfs


def unique_output_path(out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}_transcript.txt"
    n = 2
    while path.exists():
        path = out_dir / f"{stem}_transcript_{n}.txt"
        n += 1
    return path


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def chunk_instructions(chunk, prev_context: str, speakers: list[int], cfg: dict, has_pdf: bool) -> str:
    tcfg = cfg["transcription"]
    timestamp = cfg["output"]["timestamp"]
    lines = [
        f"Transcribe the whole audio clip given at the end ({chunk.end - chunk.start:.0f} seconds), "
        "from its first word to its last word."
    ]
    if has_pdf:
        lines.append("The presentation material given above is reference context only.")

    if chunk.index > 0 and prev_context:
        lines += [
            "",
            "The recording is processed in consecutive clips. The end of the transcript so far is:",
            "---",
            prev_context,
            "---",
            f"This clip starts {chunk.overlap:.0f} seconds before the end of that transcript, so its beginning "
            "repeats it. Transcribe the whole clip anyway; the repeated part is removed automatically. "
            "Use the transcript above only to keep speaker numbers consistent.",
        ]
        if tcfg["speaker_labels"] and speakers:
            used = ", ".join(f"Speaker {n}" for n in speakers)
            lines.append(
                f"Speakers identified so far: {used}. The same people are most likely still talking, so reuse "
                "these numbers. Use a new number only for a voice that is clearly different from all of them."
            )

    lines.append("")
    if tcfg["speaker_labels"]:
        lines += [
            "Start a new speaker turn every time the voice changes, even for very short replies "
            "such as 네, 예, 어, 아.",
            "Output format, one speaker turn at a time:",
        ]
        if timestamp:
            lines += ["[MM:SS] Speaker N:", "<what was said>",
                      "MM:SS is the time from the start of THIS clip when the turn begins."]
        else:
            lines += ["Speaker N:", "<what was said>"]
    else:
        lines.append("Do not label speakers. Output only the spoken text.")
    if not tcfg["acoustic_events"]:
        lines.append("Do not include acoustic event tags such as [웃음].")
    if not tcfg["preserve_language"]:
        lines.append("You may normalize the transcript into the dominant language of the recording.")
    lines.append("If nothing is spoken in this clip, output nothing.")
    return "\n".join(lines)


def turn_instructions(cfg: dict, has_pdf: bool, n_clips: int) -> str:
    tcfg = cfg["transcription"]
    lines = [
        f"Below are {n_clips} consecutive speaker turns from the recording, in order. Each turn is a separate "
        "audio clip with its number and speaker. The speakers were already identified by voice analysis; keep them.",
        f"Transcribe each clip verbatim, only the words heard in that clip. Output exactly {n_clips} lines, "
        f"[1] to [{n_clips}], one per clip, in the form '[number] text', and nothing else. Never merge two clips "
        "into one line, even if they are short. If a clip has no speech, output only '[number]'.",
    ]
    if has_pdf:
        lines.append("The presentation material given above is reference context only.")
    if not tcfg["acoustic_events"]:
        lines.append("Do not include acoustic event tags such as [웃음].")
    if not tcfg["preserve_language"]:
        lines.append("You may normalize the transcript into the dominant language of the recording.")
    return "\n".join(lines)


def _context_head(base_prompt: str, pdf: PdfContext) -> list[dict]:
    head = base_prompt.strip()
    if pdf.text:
        head += "\n\nPresentation material (text extracted from the PDF):\n" + pdf.text
    content: list[dict] = [{"type": "text", "text": head}]
    if pdf.images:
        content.append({"type": "text", "text": "Presentation slides:"})
        content += [{"type": "image", "image": img} for img in pdf.images]
    return content


# 지시는 모두 오디오 앞에 둔다. 오디오 뒤의 지시는 무시되고 "네, 알겠습니다." 같은 대답이 나왔다
# (tools/diagnose_audio.py A/B 비교).
def build_content(base_prompt: str, pdf: PdfContext, chunk, instructions: str) -> list[dict]:
    content = _context_head(base_prompt, pdf)
    content.append({"type": "text", "text": instructions + "\n\nAudio:"})
    content.append({"type": "audio", "audio": chunk.samples})
    return content


def build_turn_content(base_prompt: str, pdf: PdfContext, audio, window, instructions: str) -> list[dict]:
    content = _context_head(base_prompt, pdf)
    content.append({"type": "text", "text": instructions})
    for i, seg in enumerate(window, 1):
        content.append({"type": "text", "text": f"\n[{i}] Speaker {seg.speaker}:"})
        content.append({"type": "audio", "audio": _clip(audio, seg)})
    return content


def _clip(audio, seg):
    return normalize(audio[int(max(0.0, seg.start - CLIP_PAD) * SAMPLE_RATE): int((seg.end + CLIP_PAD) * SAMPLE_RATE)])


# 묶음 전사에서 모델이 마지막 몇 개 오디오를 건너뛰는 일이 있어(29개 묶음 중 10개), 빈 오디오는 하나씩
# 다시 전사한다. 글자가 적은 오디오까지 다시 하면 옆 턴의 말이 섞여 들어와 오히려 나빠졌다
# (precision 80.2% → 74.4%).
def needs_retry(seg, text: str) -> bool:
    return not text and seg.end - seg.start >= 0.5


def retry_turn(model, base_prompt: str, pdf: PdfContext, audio, seg) -> str:
    content = _context_head(base_prompt, pdf)
    content.append({"type": "text", "text": (
        f"Transcribe the audio clip below verbatim. It is one speaker turn (Speaker {seg.speaker}). "
        "Output only the spoken text, without a speaker label or number. If there is no speech, output nothing."
        "\n\nAudio:")})
    content.append({"type": "audio", "audio": _clip(audio, seg)})
    raw = model.generate(content)
    return parse_numbered("[1] " + raw).get(1, "")


def transcribe_file(
    model,
    audio_path: Path,
    pdf_paths: list[Path],
    cfg: dict,
    base_prompt: str,
    out_dir: Path,
    log_dir: Path,
    max_chunks: int | None = None,
) -> Path:
    log.info("=" * 60)
    log.info("녹음: %s", audio_path.name)
    pdf = build_pdf_context(pdf_paths, cfg["pdf"])
    if pdf.empty:
        log.info("발표자료: 없음 (오디오만 사용)")
    else:
        log.info("발표자료: %s (텍스트 %d자, 이미지 %d장)", ", ".join(pdf.sources), len(pdf.text), len(pdf.images))

    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    max_len = cfg["audio"]["chunk_length"]
    by_speaker = cfg["diarization"]["enabled"] and cfg["transcription"]["speaker_labels"]
    if by_speaker:
        log.info("길이: %s, 화자 구분 중 (CPU, 녹음 길이의 1/5 정도 걸립니다)...", _clock(duration))
        t0 = time.time()
        segments = diarize(audio, cfg["diarization"])
        jobs = make_windows(segments, max_len)
        log.info("화자 %d명, 발화 묶음 %d개 (%.0fs)", len({s.speaker for s in segments}), len(jobs), time.time() - t0)
    else:
        jobs = split_chunks(audio, max_len, cfg["audio"]["overlap"])
        log.info("길이: %s, 구간 %d개로 나누어 처리합니다.", _clock(duration), len(jobs))
    if max_chunks:
        jobs = jobs[:max_chunks]

    out_path = unique_output_path(out_dir, audio_path.stem)
    raw_log = log_dir / f"{out_path.stem}_chunks.log"
    timestamp = cfg["output"]["timestamp"]
    context_lines = int(cfg["transcription"]["context_lines"])

    turns = []
    started = time.time()
    with open(raw_log, "w", encoding="utf-8") as raw_f:
        for i, job in enumerate(jobs, 1):
            if by_speaker:
                label = f"[{i}/{len(jobs)}] {_clock(job[0].start)}-{_clock(job[-1].end)}"
                instructions = turn_instructions(cfg, not pdf.empty, len(job))
                content = build_turn_content(base_prompt, pdf, audio, job, instructions)
            else:
                label = f"[{i}/{len(jobs)}] {_clock(job.start)}-{_clock(job.end)}"
                if cfg["audio"]["skip_silence"] and job.is_silent:
                    log.info("%s 무음 구간 건너뜀", label)
                    continue
                instructions = chunk_instructions(
                    job, context_tail(turns, context_lines), speakers_seen(turns), cfg, not pdf.empty
                )
                content = build_content(base_prompt, pdf, job, instructions)

            t0 = time.time()
            if model is None:  # --dry-run
                raw = ""
                raw_f.write(f"##### {label}\n{instructions}\n\n")
            else:
                raw = model.generate(content)
                raw_f.write(f"##### {label}\n{raw}\n\n")
            raw_f.flush()

            if by_speaker:
                texts = parse_numbered(raw)
                for k, seg in enumerate(job, 1):
                    if model is not None and needs_retry(seg, texts.get(k, "")):
                        again = retry_turn(model, base_prompt, pdf, audio, seg)
                        raw_f.write(f"## retry [{k}] {_clock(seg.start)}: {again}\n\n")
                        if len(again) > len(texts.get(k, "")):
                            texts[k] = again
                    if texts.get(k):
                        append_turns(turns, [Turn(seg.start, seg.speaker, [texts[k]])])
            else:
                new_turns = parse_output(raw, job.start, job.end - job.start)
                append_turns(turns, remove_overlap(turns, new_turns))

            # 구간마다 결과 파일을 갱신해 중간에 멈춰도 그때까지의 결과가 남도록 한다.
            out_path.write_text(render(turns, timestamp), encoding="utf-8")

            elapsed = time.time() - started
            remaining = elapsed / i * (len(jobs) - i)
            log.info("%s 완료 (%.1fs, 남은 시간 약 %s)", label, time.time() - t0, _clock(remaining))

    if not out_path.exists():
        out_path.write_text("", encoding="utf-8")
    log.info("결과 저장: %s (총 %s 소요)", out_path, _clock(time.time() - started))
    if model is not None and (vram := model.vram_report()):
        log.info(vram)
    return out_path
