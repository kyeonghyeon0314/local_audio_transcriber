"""전사 결과를 기준 전사(예: Gemini 결과)와 비교한다.

화자 표기/시간/마크다운/음향 태그를 빼고 글자 단위로 맞춰 본다.
  precision: 우리 결과 중 기준에도 있는 글자 비율 (앞 N구간만 돌린 부분 결과도 의미 있음)
  recall:    기준 중 우리 결과에도 있는 글자 비율 (전체를 돌렸을 때 의미 있음)

사용법: python tools/compare.py output/xxx_transcript.txt Gemini_answer.txt
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

SPEAKER_RE = re.compile(r"(\[[\d:]+\]\s*)?\**\s*(Speaker|화자)\s*\d+\s*\**\s*[:：]\s*\**", re.I)
TAG_RE = re.compile(r"\[[^\]]*\]|\$|\*")


def normalize(text: str) -> str:
    text = TAG_RE.sub("", SPEAKER_RE.sub("", text))
    return re.sub(r"[\W_]+", "", text).lower()


def turns(text: str) -> int:
    return len(SPEAKER_RE.findall(text))


def compare(ours: str, ref: str) -> dict:
    a, b = normalize(ours), normalize(ref)
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return {
        "chars": (len(a), len(b)),
        "turns": (turns(ours), turns(ref)),
        "precision": matched / len(a) if a else 0.0,
        "recall": matched / len(b) if b else 0.0,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ours, ref = (Path(p).read_text(encoding="utf-8") for p in sys.argv[1:3])
    r = compare(ours, ref)
    print(f"글자 수   결과 {r['chars'][0]} / 기준 {r['chars'][1]}")
    print(f"화자 턴   결과 {r['turns'][0]} / 기준 {r['turns'][1]}")
    print(f"precision {r['precision']:.1%}  recall {r['recall']:.1%}")


if __name__ == "__main__":
    a = "**Speaker 1**: 여기 MPCC에서 [웃음] 에러가"
    b = "[00:00:03] Speaker 2:\n여기 MPCC에서 에러가"
    assert normalize(a) == normalize(b) == "여기mpcc에서에러가", normalize(a)
    assert turns(a) == turns(b) == 1
    assert compare(a, b)["precision"] == 1.0
    if len(sys.argv) >= 3:
        main()
