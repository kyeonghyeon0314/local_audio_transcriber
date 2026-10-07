import numpy as np

from transcriber.audio import SAMPLE_RATE, split_chunks
from transcriber.transcript import (
    append_turns,
    context_tail,
    parse_output,
    remove_overlap,
    render,
)


def test_parse_output_formats():
    raw = (
        "[00:03] Speaker 1:\n이번에는 MPCC 결과를 설명드리겠습니다.\n"
        "**Speaker 2:** 그러면 여기 Q 값은 고정한 건가요?\n"
        "[웃음]\n"
    )
    turns = parse_output(raw, chunk_start=100.0, chunk_duration=25.0)
    assert [t.speaker for t in turns] == [1, 2]
    assert turns[0].time == 103.0
    assert turns[1].time == 100.0  # 타임스탬프가 없으면 구간 시작 시각
    assert turns[1].lines == ["그러면 여기 Q 값은 고정한 건가요?", "[웃음]"]


def test_parse_output_ignores_out_of_range_timestamp():
    turns = parse_output("[12:00] Speaker 1:\n안녕하세요", chunk_start=40.0, chunk_duration=25.0)
    assert turns[0].time == 40.0


def test_remove_overlap_drops_repeated_turn_and_trims_prefix():
    prev = parse_output(
        "[00:00] Speaker 1:\n여기서는 Model Predictive Control을 이용해서 lateral acceleration constraint를 걸었습니다.\n"
        "[00:15] Speaker 2:\n그러면 Q 값은 고정한 건가요?",
        0.0, 25.0,
    )
    new = parse_output(
        "[00:00] Speaker 2:\n그러면 Q 값은 고정한 건가요?\n"
        "[00:03] Speaker 1:\n네. Q는 고정하고 R 값만 변경했습니다.",
        20.0, 25.0,
    )
    result = remove_overlap(prev, new, overlap_end=25.0)
    assert [t.speaker for t in result] == [1]

    new = parse_output("[00:00] Speaker 2:\n고정한 건가요? 그리고 R은요?", 20.0, 25.0)
    result = remove_overlap(prev, new, overlap_end=25.0)
    assert result[0].lines == ["그리고 R은요?"]


def test_remove_overlap_keeps_new_speech():
    prev = parse_output("[00:00] Speaker 1:\n첫 번째 문장입니다.", 0.0, 25.0)
    new = parse_output("[00:06] Speaker 2:\n완전히 새로운 질문이 있습니다.", 20.0, 25.0)
    assert remove_overlap(prev, new, overlap_end=25.0) == new


def test_append_and_render():
    turns = parse_output("[00:00] Speaker 1:\n앞부분", 0.0, 25.0)
    append_turns(turns, parse_output("[00:05] Speaker 1:\n이어서 말함\n[00:10] Speaker 2:\n질문", 20.0, 25.0))
    assert render(turns, timestamp=True) == (
        "[00:00:00] Speaker 1:\n앞부분\n이어서 말함\n[00:00:30] Speaker 2:\n질문\n"
    )
    assert render(turns, timestamp=False).startswith("Speaker 1:\n")
    assert context_tail(turns, 2) == "Speaker 2:\n질문"


def test_split_chunks_covers_whole_recording():
    for seconds in (5, 25, 26, 47, 70, 3601):
        audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        chunks = split_chunks(audio, 25, 5)
        assert chunks[0].start == 0 and chunks[0].overlap == 0
        assert abs(chunks[-1].end - seconds) < 1e-6
        for a, b in zip(chunks, chunks[1:]):
            assert b.start < a.end  # 빈틈 없이 겹침
            assert abs(b.overlap - (a.end - b.start)) < 1e-6
            assert b.end - b.start <= 25
        assert all(c.is_silent for c in chunks)
