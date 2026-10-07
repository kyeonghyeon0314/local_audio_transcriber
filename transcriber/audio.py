"""오디오 디코딩과 구간 분할.

STT를 거치지 않고, 잘라낸 원본 파형(16kHz mono float32)을 그대로 모델에 넘긴다.
디코딩은 imageio-ffmpeg에 포함된 ffmpeg 실행 파일을 사용하므로 별도 설치가 필요 없다.
"""

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
SILENCE_RMS = 1e-3  # 약 -60 dBFS
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm", ".mp4", ".mkv", ".mov"}


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("ffmpeg를 찾을 수 없습니다. setup.bat을 다시 실행해 주세요.")


def load_audio(path) -> np.ndarray:
    cmd = [
        _ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"오디오 디코딩 실패: {path}\n{err}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"오디오가 비어 있습니다: {path}")
    return audio


@dataclass
class Chunk:
    index: int
    start: float  # 초, 전체 녹음 기준
    end: float
    overlap: float  # 앞 구간과 겹치는 길이(초). 첫 구간은 0
    samples: np.ndarray
    is_silent: bool


def split_chunks(audio: np.ndarray, chunk_length: float, overlap: float) -> list[Chunk]:
    total = len(audio) / SAMPLE_RATE
    step = chunk_length - overlap
    starts = [0.0]
    while starts[-1] + chunk_length < total:
        starts.append(starts[-1] + step)
    # 마지막 구간은 끝에 맞춰 앞으로 당겨 항상 충분한 길이를 갖게 한다(겹침이 늘어남).
    if len(starts) > 1:
        starts[-1] = max(0.0, total - chunk_length)

    chunks = []
    for i, start in enumerate(starts):
        end = min(start + chunk_length, total)
        prev_end = starts[i - 1] + chunk_length if i else start
        segment = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        silent = segment.size == 0 or float(np.sqrt(np.mean(segment**2))) < SILENCE_RMS
        chunks.append(Chunk(i, start, end, max(0.0, prev_end - start), _normalize(segment), silent))
    return chunks


def _normalize(segment: np.ndarray, target_peak: float = 0.9, max_gain: float = 10.0) -> np.ndarray:
    """조용한 녹음을 키워 준다. 이미 충분히 큰 신호는 건드리지 않는다."""
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    if peak <= 0:
        return segment
    gain = min(target_peak / peak, max_gain)
    return (segment * gain).astype(np.float32) if gain > 1.0 else segment
