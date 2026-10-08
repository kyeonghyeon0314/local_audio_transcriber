"""Local Multimodal Transcriber

사용법:
  1) input 폴더에 녹음 파일과 발표자료 PDF를 넣고 run.bat 실행
  2) 또는 녹음/PDF 파일을 run.bat 위로 끌어다 놓기
결과는 output 폴더에 <녹음파일명>_transcript.txt 로 저장된다.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "accelerate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="녹음 + 발표자료 PDF → 화자 구분 전사 TXT")
    parser.add_argument("files", nargs="*", help="녹음 파일/PDF (생략하면 input 폴더 사용)")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--prompt", default=str(ROOT / "prompt.txt"))
    parser.add_argument("--output", default=str(ROOT / "output"))
    parser.add_argument("--no-pdf", action="store_true", help="PDF 없이 오디오만 사용 (비교 테스트용)")
    parser.add_argument("--max-chunks", type=int, default=None, help="앞에서부터 N개 구간만 처리 (빠른 테스트용)")
    parser.add_argument("--dry-run", action="store_true", help="모델 없이 입력/구간/프롬프트만 확인")
    parser.add_argument("--no-open", action="store_true", help="완료 후 output 폴더를 열지 않음")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_file = setup_logging(log_dir)
    log = logging.getLogger("run")

    from transcriber.config import load_config
    from transcriber.pipeline import find_inputs, pick_pdfs, transcribe_file

    cfg = load_config(Path(args.config))
    base_prompt = Path(args.prompt).read_text(encoding="utf-8")

    input_dir = ROOT / "input"
    input_dir.mkdir(exist_ok=True)
    sources = [Path(f) for f in args.files] or [input_dir]
    audios, pdfs = find_inputs(sources)
    if args.no_pdf:
        pdfs = []

    if not audios:
        log.info("처리할 녹음 파일이 없습니다.")
        log.info("input 폴더에 녹음 파일(wav/mp3/m4a 등)과 발표자료 PDF를 넣고 다시 실행해 주세요.")
        if not args.files:
            open_folder(input_dir)
        return 1

    log.info("녹음 %d개, PDF %d개를 찾았습니다.", len(audios), len(pdfs))

    model = None
    if not args.dry_run:
        from transcriber.model import GemmaTranscriber

        try:
            model = GemmaTranscriber(cfg)
        except Exception as e:
            log.error("모델을 불러오지 못했습니다: %s", e)
            log.error(
                "- 처음 실행이라면 setup.bat에서 Hugging Face 로그인과 Gemma 라이선스 동의를 했는지 확인해 주세요.\n"
                "- 자세한 내용: %s", log_file,
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            return 2

    results, failures = [], []
    for audio in audios:
        try:
            results.append(
                transcribe_file(model, audio, pick_pdfs(audio, pdfs), cfg, base_prompt, out_dir, log_dir, args.max_chunks)
            )
        except KeyboardInterrupt:
            log.warning("사용자가 중단했습니다. 지금까지의 결과는 output 폴더에 저장되어 있습니다.")
            break
        except Exception as e:
            failures.append(audio)
            log.error("실패: %s (%s)", audio.name, e)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    log.info("=" * 60)
    log.info("완료: %d개, 실패: %d개", len(results), len(failures))
    for path in results:
        log.info("  %s", path.name)
    if failures:
        log.info("오류 로그: %s", log_file)
    if results and cfg["output"]["open_folder"] and not args.no_open:
        open_folder(out_dir)
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(main())
