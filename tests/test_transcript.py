import numpy as np

from transcriber.audio import SAMPLE_RATE, split_chunks
from transcriber.diarize import SpeakerSegment, make_windows, number_by_appearance, resolve_overlaps
from transcriber.transcript import (
    append_turns,
    context_tail,
    parse_numbered,
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
    result = remove_overlap(prev, new)
    assert [t.speaker for t in result] == [1]

    new = parse_output("[00:00] Speaker 2:\n고정한 건가요? 그리고 R은요?", 20.0, 25.0)
    result = remove_overlap(prev, new)
    assert result[0].lines == ["그리고 R은요?"]


def test_parse_output_accepts_absolute_timestamp():
    turns = parse_output("[01:05] Speaker 1:\n안녕하세요", chunk_start=60.0, chunk_duration=25.0)
    assert turns[0].time == 65.0


def test_remove_overlap_real_chunks():
    # 실제 녹음에서 모델이 겹친 5초를 다시 전사한 사례 (앞 구간 끝 단어가 잘려 있음)
    prev = parse_output(
        "[00:40] Speaker 1: 다른 이제 VSD 논문 걸로 바꾸니까 사라지기는 했는데 그거는 어 어떻게 해결은 안 되는 거 같기도 하고 근데 이",
        40.0, 25.0,
    )
    new = parse_output(
        "[00:00] Speaker 2: 어, 어떻게 해결은 안 되는 거 같기도 하고. 근데 이게 속도 이렇게 나누잖아. 네.",
        60.0, 25.0,
    )
    assert render(remove_overlap(prev, new), False) == "Speaker 2:\n이게 속도 이렇게 나누잖아. 네.\n"
    assert prev[-1].lines[-1].endswith("근데")  # 잘린 "이"는 앞 구간에서 지움


def test_remove_overlap_keeps_new_speech():
    prev = parse_output("[00:00] Speaker 1:\n첫 번째 문장입니다.", 0.0, 25.0)
    new = parse_output("[00:06] Speaker 2:\n완전히 새로운 질문이 있습니다.", 20.0, 25.0)
    assert remove_overlap(prev, new) == new


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


def test_parse_output_collapses_runaway_repetition():
    raw = "[00:01] Speaker 1: 지금 이거 또 " + "그... " * 50 + "끝\n[00:05] Speaker 2: 네. 네."
    turns = parse_output(raw, 0.0, 25.0)
    assert turns[0].lines == ["지금 이거 또 그... 끝"]
    assert turns[1].lines == ["네. 네."]  # 짧은 반복은 그대로


def test_parse_numbered():
    raw = "[1] Speaker 2: 어, 해결은 안 되는 거 같기도 하고.\n[2] **Speaker 1:** 근데\n이어서\n[3]\n"
    assert parse_numbered(raw) == {1: "어, 해결은 안 되는 거 같기도 하고.", 2: "근데 이어서", 3: ""}
    # 실제 출력: 여러 번호를 한 줄에 몰아 씀
    assert parse_numbered("[1] 아 아 [2] Speaker 2: 네 [3] Speaker 1: 그럼 [웃음]") == {
        1: "아 아", 2: "네", 3: "그럼 [웃음]"}


def test_resolve_overlaps_gives_interjection_to_short_speaker():
    S = SpeakerSegment
    out = resolve_overlaps([S(37.8, 51.5, 2), S(40.9, 42.5, 1), S(51.0, 55.0, 1)])
    assert [(p.start, p.end, p.speaker) for p in out] == [
        (37.8, 40.9, 2), (40.9, 42.5, 1), (42.5, 51.0, 2), (51.0, 55.0, 1)]


def test_number_by_appearance():
    segs = number_by_appearance([SpeakerSegment(0, 1, 5), SpeakerSegment(1, 2, 0), SpeakerSegment(2, 3, 5)])
    assert [s.speaker for s in segs] == [1, 2, 1]


def test_make_windows():
    S = SpeakerSegment
    segs = [S(0.0, 4.0, 1), S(4.5, 8.0, 1),  # 쉼 0.5초 → 한 턴
            S(8.0, 8.5, 2),  # 짧은 대답
            S(9.0, 69.0, 1)]  # 60초 독백 → 25초 이하 조각 3개
    windows = make_windows(segs, 25.0)
    pieces = [p for w in windows for p in w]
    assert (pieces[0].start, pieces[0].end, pieces[0].speaker) == (0.0, 8.0, 1)
    assert pieces[1].speaker == 2
    assert [round(p.end - p.start) for p in pieces[2:]] == [20, 20, 20]
    assert all(sum(p.end - p.start for p in w) <= 25.0 for w in windows)
    assert len(windows) == 4  # [턴1+대답], [조각1], [조각2], [조각3]
