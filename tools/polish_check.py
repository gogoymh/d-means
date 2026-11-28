"""Flag Korean writing-polish violations in agent messages before they are posted.

Rules come from the project polish skill (DM-28, DM-30, DM-32):
  particle   doubled particles such as '을(를)' or '(으)로'; suggests the one that fits
  register   plain '~다.' endings mixed into a message written with '~니다.'
  english    English status words such as 'done' or 'is done' inside a Korean message

Text inside quotes, backticks and '(작업: ...)' titles is left alone, since it
quotes something else on purpose.

Usage:
  python tools/polish_check.py "미션 DM-27을(를) 마쳤습니다."   # one message per argument
  python tools/polish_check.py < messages.txt                   # one message per line
Lines shaped like '[id] date time author: text' (Preconquer message_read output)
are reported by message id. Exit status is 1 when anything is flagged.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

LOG = logging.getLogger("polish_check")

# Final consonant class of the last digit as read in Korean: 0 영, 1 일, 2 이, ...
_DIGIT_CODA = {"0": "other", "1": "rieul", "2": "none", "3": "other", "4": "none",
               "5": "none", "6": "other", "7": "rieul", "8": "rieul", "9": "none"}

# Doubled form -> (particle after a coda, particle after no coda); 로 also follows ㄹ.
_PARTICLES = {
    "을(를)": ("을", "를"), "를(을)": ("을", "를"),
    "이(가)": ("이", "가"), "가(이)": ("이", "가"),
    "은(는)": ("은", "는"), "는(은)": ("은", "는"),
    "과(와)": ("과", "와"), "와(과)": ("과", "와"),
    "(으)로": ("으로", "로"), "(이)라": ("이라", "라"), "(이)나": ("이나", "나"),
}
_PARTICLE_RE = re.compile("|".join(re.escape(p) for p in _PARTICLES))

_STATUS_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:(?:is|are|was|were|has been)\s+)?"
    r"(?:done|finished|completed|merged|fixed|ready|closed|blocked|pending|in progress|wip|lgtm)"
    r"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_MASK_RE = re.compile(r"`[^`]*`|'[^'\n]*'|\"[^\"\n]*\"|‘[^’]*’|“[^”]*”|\(작업:[^)]*\)")
_SENTENCE_END_RE = re.compile(r"([가-힣]+)[.!:](?=\s|$)")
_HANGUL_RE = re.compile(r"[가-힣]")
_HEADER_RE = re.compile(r"^\[(\d+)\] \S+ \S+ [^:]+: ")


@dataclass(frozen=True)
class Violation:
    col: int  # 1-based character offset in the message
    rule: str
    text: str
    hint: str

    def format(self, label: str) -> str:
        return f"{label}:{self.col}: {self.rule} '{self.text}' -> {self.hint}"


def _coda(ch: str) -> str | None:
    """'none', 'rieul' or 'other' for the sound before a particle; None when unknown."""
    if ch in _DIGIT_CODA:
        return _DIGIT_CODA[ch]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return "none" if jong == 0 else "rieul" if jong == 8 else "other"
    return None


def _mask(text: str) -> str:
    """Blank out quoted spans, keeping offsets, so quoted examples are not flagged."""
    return _MASK_RE.sub(lambda m: " " * len(m.group()), text)


def _particle(text: str, masked: str) -> list[Violation]:
    out = []
    for m in _PARTICLE_RE.finditer(masked):
        doubled = m.group()
        with_coda, without = _PARTICLES[doubled]
        rest = masked[m.end():].lstrip()
        if doubled == "(으)로" and rest.startswith("맡"):
            with_coda, without = "을", "를"  # '~(으)로 맡았습니다' reads as '~를 맡았습니다'
        coda = _coda(text[m.start() - 1]) if m.start() else None
        if coda is None:
            hint = f"pick one of '{with_coda}' or '{without}'"
        else:
            use_coda = coda == "other" or (coda == "rieul" and not with_coda.endswith("로"))
            word = text[:m.start()].split()[-1]
            hint = f"'{word}{with_coda if use_coda else without}'"
        out.append(Violation(m.start() + 1, "particle", doubled, hint))
    return out


def _register(masked: str) -> list[Violation]:
    endings = list(_SENTENCE_END_RE.finditer(masked))
    if not any(m.group(1).endswith("니다") for m in endings):
        return []
    return [
        Violation(m.start(1) + 1, "register", m.group(1), "end with '~니다' like the rest of the message")
        for m in endings
        if m.group(1).endswith("다") and not m.group(1).endswith("니다")
    ]


def _english(masked: str) -> list[Violation]:
    if not _HANGUL_RE.search(masked):
        return []
    return [Violation(m.start() + 1, "english", m.group(), "say it in Korean, such as '끝났습니다'")
            for m in _STATUS_RE.finditer(masked)]


def check(text: str) -> list[Violation]:
    """All violations in one message, ordered by position."""
    masked = _mask(text)
    found = _particle(text, masked) + _register(masked) + _english(masked)
    return sorted(found, key=lambda v: v.col)


def split_header(line: str, default: str) -> tuple[str, str]:
    """Strip a '[id] date time author: ' prefix; returns (label, message)."""
    m = _HEADER_RE.match(line)
    return (f"#{m.group(1)}", line[m.end():]) if m else (default, line)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if argv:
        items = [(f"arg{i}", a) for i, a in enumerate(argv, 1)]
    else:
        items = [split_header(line.rstrip("\n"), f"line{i}")
                 for i, line in enumerate(sys.stdin, 1) if line.strip()]
    total = 0
    for label, message in items:
        for v in check(message):
            print(v.format(label))
            total += 1
    LOG.info("checked %d messages, found %d violations", len(items), total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
