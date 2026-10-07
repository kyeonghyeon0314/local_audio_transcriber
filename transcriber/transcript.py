"""모델 출력 파싱, 구간 간 중복 제거/병합, TXT 렌더링."""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

HEADER_RE = re.compile(
    r"^\s*(?:\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?"
    r"\**\s*(?:Speaker|SPEAKER|speaker|화자)\s*(?P<num>\d+)\s*\**\s*[:：]\s*\**\s*(?P<rest>.*)$"
)
TS_ONLY_RE = re.compile(r"^\s*\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<rest>.*)$")
EVENT_RE = re.compile(r"\[[^\]]*\]")
NON_WORD_RE = re.compile(r"[^\w]+")


@dataclass
class Turn:
    time: float  # 전체 녹음 기준 초
    speaker: int | None  # Speaker 번호, 화자 표기가 없는 줄은 None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.lines)


def _parse_ts(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def parse_output(raw: str, chunk_start: float, chunk_duration: float) -> list[Turn]:
    def to_abs(ts: str | None) -> float:
        if ts:
            rel = _parse_ts(ts)
            if rel <= chunk_duration + 2:
                return chunk_start + rel
        return chunk_start

    turns: list[Turn] = []
    for line in raw.replace("\r", "").split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        m = HEADER_RE.match(line)
        if m:
            turn = Turn(to_abs(m["ts"]), int(m["num"]))
            if m["rest"].strip():
                turn.lines.append(m["rest"].strip())
            turns.append(turn)
            continue
        m = TS_ONLY_RE.match(line)
        if m and m["rest"]:
            line = m["rest"].strip()
        if not turns:
            turns.append(Turn(chunk_start, None))
        turns[-1].lines.append(line)
    return [t for t in turns if t.lines or t.speaker is not None]


def _norm_words(text: str) -> list[str]:
    text = EVENT_RE.sub(" ", text).lower()
    return [w for w in (NON_WORD_RE.sub("", tok) for tok in text.split()) if w]


def _drop_leading_words(text: str, k: int) -> str:
    tokens = text.split()
    seen = 0
    for i, tok in enumerate(tokens):
        if NON_WORD_RE.sub("", EVENT_RE.sub("", tok)):
            seen += 1
            if seen == k:
                return " ".join(tokens[i + 1:])
    return ""


def remove_overlap(prev: list[Turn], new: list[Turn], overlap_end: float) -> list[Turn]:
    """새 구간 앞부분 중 이전 구간에서 이미 전사된 발화를 제거한다.

    overlap_end(전체 녹음 기준 초) 이전에 시작한 발화만 중복 후보로 본다.
    """
    if not prev or not new:
        return new
    prev_words = [w for t in prev[-4:] for w in _norm_words(t.text)][-120:]
    prev_joined = "".join(prev_words)
    if not prev_joined:
        return new

    result = list(new)
    while result and result[0].time <= overlap_end + 1.0:
        turn = result[0]
        words = _norm_words(turn.text)
        if not words:
            break
        joined = "".join(words)
        match = SequenceMatcher(None, joined, prev_joined, autojunk=False).find_longest_match(
            0, len(joined), 0, len(prev_joined)
        )
        if len(joined) >= 2 and match.size >= 0.8 * len(joined):
            result.pop(0)  # 통째로 이미 전사된 발화
            continue
        # 앞부분 일부만 겹치는 경우: 이전 전사의 끝 단어들과 일치하는 만큼 잘라낸다.
        best = 0
        for k in range(min(len(words), len(prev_words)), 1, -1):
            if prev_words[-k:] == words[:k]:
                best = k
                break
        if best:
            trimmed = _drop_leading_words(turn.text, best)
            result[0] = Turn(turn.time, turn.speaker, [trimmed] if trimmed else [])
        break
    return [t for t in result if t.lines]


def append_turns(all_turns: list[Turn], new: list[Turn]) -> None:
    for i, turn in enumerate(new):
        # 구간 경계에서 같은 화자의 발화가 이어지면 한 턴으로 합친다.
        if i == 0 and all_turns and all_turns[-1].speaker == turn.speaker:
            all_turns[-1].lines.extend(turn.lines)
        else:
            all_turns.append(turn)


def speakers_seen(turns: list[Turn]) -> list[int]:
    return sorted({t.speaker for t in turns if t.speaker is not None})


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(turns: list[Turn], timestamp: bool) -> str:
    out = []
    for t in turns:
        if t.speaker is not None:
            prefix = f"[{_fmt_time(t.time)}] " if timestamp else ""
            out.append(f"{prefix}Speaker {t.speaker}:")
        out.extend(t.lines)
    return "\n".join(out) + ("\n" if out else "")


def context_tail(turns: list[Turn], n_lines: int) -> str:
    """다음 구간 프롬프트에 넣을 이전 전사 끝부분. 화자 표기가 잘리지 않도록 턴 단위로 자른다."""
    if n_lines <= 0:
        return ""
    picked: list[Turn] = []
    count = 0
    for turn in reversed(turns):
        if count >= n_lines:
            break
        picked.insert(0, turn)
        count += len(turn.lines) + (turn.speaker is not None)
    text = render(picked, timestamp=False).rstrip("\n")
    lines = text.split("\n")
    # 한 턴이 매우 길면 앞쪽 내용 줄만 생략하고 화자 표기는 유지
    if len(lines) > n_lines + 1 and picked and picked[0].speaker is not None:
        lines = [lines[0], "..."] + lines[-n_lines:]
    return "\n".join(lines)
