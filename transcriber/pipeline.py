"""녹음 파일 하나를 구간별로 전사하고 결과 TXT를 만든다."""

import logging
import time
from pathlib import Path

from .audio import AUDIO_EXTENSIONS, load_audio, split_chunks, SAMPLE_RATE
from .pdf_context import PdfContext, build_pdf_context
from .transcript import (
    append_turns,
    context_tail,
    parse_output,
    remove_overlap,
    render,
    speakers_seen,
)

log = logging.getLogger(__name__)


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
        f"Transcribe the audio clip above. It covers {_clock(chunk.start)}-{_clock(chunk.end)} "
        f"of a longer recording ({chunk.end - chunk.start:.0f} seconds)."
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
            f"The first {chunk.overlap:.0f} seconds of this clip overlap with the end of the previous clip "
            "and are already transcribed above. Do not repeat them; start from the first words spoken after that.",
        ]
        if tcfg["speaker_labels"] and speakers:
            used = ", ".join(f"Speaker {n}" for n in speakers)
            lines.append(
                f"Speakers identified so far: {used}. Keep the same numbering: a voice that already appeared "
                f"keeps its number, and a new voice gets the next unused number (Speaker {max(speakers) + 1})."
            )

    lines.append("")
    if tcfg["speaker_labels"]:
        lines.append("Output format, one speaker turn at a time:")
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


def build_content(base_prompt: str, pdf: PdfContext, chunk, instructions: str) -> list[dict]:
    head = base_prompt.strip()
    if pdf.text:
        head += "\n\nPresentation material (text extracted from the PDF):\n" + pdf.text
    content: list[dict] = [{"type": "text", "text": head}]
    if pdf.images:
        content.append({"type": "text", "text": "Presentation slides:"})
        content += [{"type": "image", "image": img} for img in pdf.images]
    content.append({"type": "text", "text": "Audio:"})
    content.append({"type": "audio", "audio": chunk.samples})
    content.append({"type": "text", "text": instructions})
    return content


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
    chunks = split_chunks(audio, cfg["audio"]["chunk_length"], cfg["audio"]["overlap"])
    if max_chunks:
        chunks = chunks[:max_chunks]
    log.info("길이: %s, 구간 %d개로 나누어 처리합니다.", _clock(duration), len(chunks))

    out_path = unique_output_path(out_dir, audio_path.stem)
    raw_log = log_dir / f"{out_path.stem}_chunks.log"
    timestamp = cfg["output"]["timestamp"]
    context_lines = int(cfg["transcription"]["context_lines"])

    turns = []
    started = time.time()
    with open(raw_log, "w", encoding="utf-8") as raw_f:
        for i, chunk in enumerate(chunks, 1):
            label = f"[{i}/{len(chunks)}] {_clock(chunk.start)}-{_clock(chunk.end)}"
            if cfg["audio"]["skip_silence"] and chunk.is_silent:
                log.info("%s 무음 구간 건너뜀", label)
                continue

            instructions = chunk_instructions(
                chunk, context_tail(turns, context_lines), speakers_seen(turns), cfg, not pdf.empty
            )
            content = build_content(base_prompt, pdf, chunk, instructions)

            t0 = time.time()
            if model is None:  # --dry-run
                raw = ""
                raw_f.write(f"##### {label}\n{instructions}\n\n")
            else:
                raw = model.generate(content)
                raw_f.write(f"##### {label}\n{raw}\n\n")
            raw_f.flush()

            new_turns = parse_output(raw, chunk.start, chunk.end - chunk.start)
            new_turns = remove_overlap(turns, new_turns, chunk.start + chunk.overlap)
            append_turns(turns, new_turns)

            # 구간마다 결과 파일을 갱신해 중간에 멈춰도 그때까지의 결과가 남도록 한다.
            out_path.write_text(render(turns, timestamp), encoding="utf-8")

            elapsed = time.time() - started
            remaining = elapsed / i * (len(chunks) - i)
            log.info("%s 완료 (%.1fs, 남은 시간 약 %s)", label, time.time() - t0, _clock(remaining))

    if not out_path.exists():
        out_path.write_text("", encoding="utf-8")
    log.info("결과 저장: %s (총 %s 소요)", out_path, _clock(time.time() - started))
    return out_path
