"""모델 출력 파싱, 구간 간 중복 제거/병합, TXT 렌더링."""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

HEADER_RE = re.compile(
    r"^\s*(?:\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?"
    r"\**\s*(?:Speaker|SPEAKER|speaker|화자)\s*(?P<num>\d+)\s*\**\s*[:：]\s*\**\s*(?P<rest>.*)$"
)
# "[3] Speaker 1:" 같은 번호 표시. 모델이 여러 번호를 한 줄에 몰아 쓰기도 해서 줄 중간에서도 찾는다.
NUMBER_MARK_RE = re.compile(r"\[(\d+)\]\s*(?:\**\s*(?:Speaker|SPEAKER|speaker|화자)\s*\d+\s*\**\s*[:：]\s*\**)?")
TS_ONLY_RE = re.compile(r"^\s*\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<rest>.*)$")
EVENT_RE = re.compile(r"\[[^\]]*\]")
NON_WORD_RE = re.compile(r"[^\w]+")
# 모델이 같은 말을 끝없이 되풀이하는 경우(예: "그... 그... 그...")를 한 번으로 줄인다. 1~4단어가 4번 이상 연속일 때만.
REPEAT_RE = re.compile(r"(\S+(?:\s+\S+){0,3}?)(?:\s+\1){3,}")


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
            t = _parse_ts(ts)
            # 모델이 구간 기준(00:12)과 녹음 전체 기준(01:32)을 섞어 쓰므로 둘 다 받아 준다.
            if chunk_start - 2 <= t <= chunk_start + chunk_duration + 2:
                return t
            if t <= chunk_duration + 2:
                return chunk_start + t
        return chunk_start

    turns: list[Turn] = []
    raw = REPEAT_RE.sub(r"\1", raw)
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


def parse_numbered(raw: str) -> dict[int, str]:
    """'[번호] 텍스트' 형식의 출력을 {번호: 텍스트}로. 번호 다음 줄들은 그 번호에 이어 붙인다."""
    text = " ".join(line.strip() for line in REPEAT_RE.sub(r"\1", raw).splitlines() if not line.startswith("```"))
    parts = NUMBER_MARK_RE.split(text)  # [번호 앞 텍스트, 번호1, 텍스트1, 번호2, 텍스트2, ...]
    out: dict[int, str] = {}
    for n, body in zip(parts[1::2], parts[2::2]):
        out[int(n)] = " ".join(body.split())
    return out


def _norm_words(text: str) -> list[str]:
    text = EVENT_RE.sub(" ", text).lower()
    return [w for w in (NON_WORD_RE.sub("", tok) for tok in text.split()) if w]


# 구간 경계 중복 제거 파라미터 (정규화된 글자 수 기준)
TAIL_CHARS = 150  # 이전 전사 끝에서 비교할 길이
HEAD_CHARS = 200  # 새 구간 앞에서 비교할 길이 (5초 겹침이면 보통 30~60자)
MIN_MATCH = 6  # 이보다 짧게 겹치면 우연의 일치로 본다
END_SLACK = 12  # 이전 전사 끝에서 이만큼 떨어진 곳까지는 겹침으로 인정 (끝부분 오인식 허용)


def remove_overlap(prev: list[Turn], new: list[Turn]) -> list[Turn]:
    """새 구간 앞부분 중 이전 구간에서 이미 전사된 부분을 잘라낸다.

    모델에게 겹친 부분을 건너뛰라고 하면 들쭉날쭉하게 따르므로, 구간 전체를 전사하게 한 뒤
    이전 전사의 끝과 새 전사의 앞을 글자 단위로 정렬해 겹친 만큼 버린다.
    """
    if not prev or not new:
        return new
    tail = "".join(_norm_words(" ".join(t.text for t in prev[-4:])))[-TAIL_CHARS:]
    tokens = [(i, tok) for i, t in enumerate(new) for tok in t.text.split()]
    ends, pos = [], 0
    for _, tok in tokens:
        pos += len("".join(_norm_words(tok)))
        ends.append(pos)
    head = "".join("".join(_norm_words(tok)) for _, tok in tokens)[:HEAD_CHARS]

    blocks = [m for m in SequenceMatcher(None, tail, head, autojunk=False).get_matching_blocks() if m.size >= MIN_MATCH]
    if not blocks:
        return new
    last = max(blocks, key=lambda m: m.a + m.size)
    if last.a + last.size < len(tail) - END_SLACK:
        return new
    cut = last.b + last.size
    drop = sum(1 for e in ends if e <= cut)  # 끝까지 겹친 단어만 버린다(걸친 단어는 남김)
    if drop < len(ends) and cut > (ends[drop - 1] if drop else 0) and last.a + last.size == len(tail):
        # 앞 구간이 단어 중간에서 잘렸다(예: "찾" / "찾은"). 온전한 새 단어를 남기고 앞 구간의 조각을 지운다.
        words = prev[-1].lines[-1].split()
        if len(words) > 1:
            prev[-1].lines[-1] = " ".join(words[:-1])
        else:
            prev[-1].lines.pop()

    result = []
    for i, turn in enumerate(new):
        kept = [tok for j, (ti, tok) in enumerate(tokens) if ti == i and j >= drop]
        if kept:
            result.append(Turn(turn.time, turn.speaker, [" ".join(kept)]))
    return result


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
