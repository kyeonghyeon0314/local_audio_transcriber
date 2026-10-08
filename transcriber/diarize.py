"""화자 구분(diarization) 보조 기능.

음성을 텍스트로 바꾸지 않고 "누가 언제 말했는지"만 판단한다. 전사는 여전히 Gemma가
원본 오디오 구간을 직접 듣고 한다. Gemma 4 E4B/12B 모두 목소리를 구분하지 못해
(README의 실측 참고) 화자 구분만 이 작은 모델들에 맡긴다. CPU에서 동작하며 VRAM을 쓰지 않는다.

- 발화 구간 검출: pyannote segmentation-3.0 (ONNX, MIT)
- 화자 임베딩: WeSpeaker ResNet34 (ONNX, VoxCeleb)
"""

import logging
import math
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

log = logging.getLogger(__name__)

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
# 같은 화자의 발화 사이 쉼이 이보다 짧으면 한 턴으로 합친다(초). 같은 화자 오디오가 연달아
# 따로 들어가면 Gemma가 둘을 한 항목으로 합쳐 답해 뒤 번호의 텍스트가 밀렸다.
MERGE_GAP = 3.0


@dataclass
class SpeakerSegment:
    start: float  # 초, 녹음 전체 기준
    end: float
    speaker: int  # 1부터


def ensure_models(model_dir: Path) -> tuple[Path, Path]:
    """화자 구분 모델이 없으면 내려받는다 (최초 1회, 약 33MB)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    seg = model_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb = model_dir / "wespeaker_en_voxceleb_resnet34_LM.onnx"
    if not seg.exists():
        log.info("화자 구분 모델 내려받는 중 (segmentation, 약 7MB)...")
        archive = model_dir / "segmentation.tar.bz2"
        urllib.request.urlretrieve(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(model_dir, filter="data")
        archive.unlink()
    if not emb.exists():
        log.info("화자 구분 모델 내려받는 중 (speaker embedding, 약 26MB)...")
        tmp = emb.with_suffix(".part")
        urllib.request.urlretrieve(EMBEDDING_URL, tmp)
        tmp.rename(emb)
    return seg, emb


def diarize(audio: np.ndarray, cfg: dict, model_dir: Path = MODEL_DIR) -> list[SpeakerSegment]:
    import sherpa_onnx

    seg_model, emb_model = ensure_models(model_dir)
    num_speakers = int(cfg.get("num_speakers", 0))
    threads = os.cpu_count() or 4
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1,
            threshold=float(cfg.get("threshold", 0.7)),
        ),
        min_duration_on=0.2,
        min_duration_off=0.3,
    )
    if not config.validate():
        raise RuntimeError("화자 구분 모델 설정이 올바르지 않습니다.")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    assert sd.sample_rate == SAMPLE_RATE
    result = sd.process(audio).sort_by_start_time()
    return number_by_appearance([SpeakerSegment(s.start, s.end, s.speaker) for s in result])


def number_by_appearance(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """클러스터 번호를 처음 말한 순서대로 1, 2, 3...으로 바꾼다."""
    mapping: dict[int, int] = {}
    for s in segments:
        mapping.setdefault(s.speaker, len(mapping) + 1)
    return [SpeakerSegment(s.start, s.end, mapping[s.speaker]) for s in segments]


def resolve_overlaps(segments: list[SpeakerSegment], min_piece: float = 0.2) -> list[SpeakerSegment]:
    """겹치는 화자 구간을 겹치지 않게 자른다.

    긴 발화 중간에 다른 사람이 끼어들면(예: 37.8~51.5초 화자2 안에 40.9~42.5초 화자1) 그 시간은
    짧은 쪽(끼어든 사람)에게 준다. 그래야 오디오 조각마다 한 사람 목소리만 들어가고, 같은 말이
    두 화자에게 중복으로 전사되지 않는다.
    """
    bounds = sorted({t for s in segments for t in (s.start, s.end)})
    pieces: list[SpeakerSegment] = []
    for a, b in zip(bounds, bounds[1:]):
        active = [s for s in segments if s.start <= a and s.end >= b]
        if not active:
            continue
        speaker = min(active, key=lambda s: s.end - s.start).speaker
        if pieces and pieces[-1].speaker == speaker and pieces[-1].end == a:
            pieces[-1].end = b
        else:
            pieces.append(SpeakerSegment(a, b, speaker))
    # 너무 짧은 조각은 앞 조각에 붙인다
    result: list[SpeakerSegment] = []
    for p in pieces:
        if result and p.end - p.start < min_piece and result[-1].end == p.start:
            result[-1].end = p.end
        elif result and result[-1].speaker == p.speaker and result[-1].end == p.start:
            result[-1].end = p.end
        else:
            result.append(p)
    return result


def make_windows(segments: list[SpeakerSegment], max_len: float) -> list[list[SpeakerSegment]]:
    """화자 구간을 턴으로 합치고, 오디오 길이 합이 max_len 이하가 되도록 연속된 턴끼리 묶는다.

    묶음 하나가 Gemma 호출 한 번이다. 턴마다 따로 부르면 짧은 턴("네")에서 앞 문맥을
    베껴 쓰는 문제가 있어, 번호 붙인 여러 오디오를 한 번에 넣는다.
    """
    turns: list[SpeakerSegment] = []
    for s in resolve_overlaps(segments):
        last = turns[-1] if turns else None
        if (last and last.speaker == s.speaker and s.start - last.end < MERGE_GAP
                and max(s.end, last.end) - last.start <= max_len):
            last.end = max(last.end, s.end)
        else:
            turns.append(SpeakerSegment(s.start, s.end, s.speaker))

    pieces: list[SpeakerSegment] = []
    for t in turns:  # 한 턴이 max_len보다 길면 같은 길이로 나눈다
        n = max(1, math.ceil((t.end - t.start) / max_len))
        step = (t.end - t.start) / n
        pieces += [SpeakerSegment(t.start + k * step, t.start + (k + 1) * step, t.speaker) for k in range(n)]

    windows: list[list[SpeakerSegment]] = []
    total = 0.0
    for p in pieces:
        d = p.end - p.start
        if windows and total + d <= max_len:
            windows[-1].append(p)
            total += d
        else:
            windows.append([p])
            total = d
    return windows
