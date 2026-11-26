"""Evidence checks for the DM-34 grading of DM-30 (.claude/skills/polish/SKILL.md).

Not collected by pytest on purpose: each check confirms a finding cited in the
grading at the cited file:line. Run one check by name:

    uv run python tests/judge_dm30_evidence.py frontmatter|particles|content|tests

frontmatter confirms a finding DM-32 has since fixed; the others confirm defects still open.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / ".claude/skills/polish/SKILL.md").read_text(encoding="utf-8").splitlines()
TEST = (ROOT / "tests/test_polish_skill.py").read_text(encoding="utf-8").splitlines()

# Final consonant of the last spoken digit: 0 = none, 1 = ㄹ, 2 = any other.
_CODA = {"2": 0, "4": 0, "5": 0, "9": 0, "1": 1, "7": 1, "8": 1, "0": 2, "3": 2, "6": 2}
_OK = {
    0: {"를", "는", "가", "로", "와"},
    1: {"을", "은", "이", "로", "과"},
    2: {"을", "은", "이", "으로", "과"},
}


def _line(lines: list[str], needle: str) -> int:
    """Index of the first line containing needle, or -1."""
    return next((i for i, l in enumerate(lines) if needle in l), -1)


def frontmatter() -> bool:
    # DM-30 left line 3 unquoted, so YAML cut it at " #general"; DM-32 has since quoted it.
    line = SKILL[2]
    value = line.split(":", 1)[1].strip()
    return line.startswith("description:") and "조사 병기" in value \
        and value.startswith('"') and value.endswith('"') and "#general" in value


def particles() -> bool:
    # Every rewrite that puts a particle after a ticket or plain number follows the table on lines 72-76.
    rewrites, label = [], ""
    for l in SKILL:
        if l.startswith("- "):
            label = l
        if label.startswith("- 고친 예") and "`" in l:
            rewrites.append(l)
    seen = 0
    for line in rewrites:
        for num, particle in re.findall(r"(?:DM-|#|\b)(\d+)(으로|을|를|은|는|이|가|와|과|로)(?![가-힣])", line):
            seen += 1
            if particle not in _OK[_CODA[num[-1]]]:
                print(f"mismatch: {num}{particle} in {line.strip()}")
                return False
    return seen >= 8


def content() -> bool:
    # SKILL.md:64 names DM-27/28/31, yet the section cites DM-15, DM-20 and DM-22 too.
    h = _line(SKILL, "## DM-27, DM-28, DM-31")
    nxt = _line(SKILL, "## 작성 뒤 체크리스트")
    section = "\n".join(SKILL[h:nxt])
    heading = h >= 0 and all(t in section for t in ("DM-15", "DM-20", "DM-22"))
    # SKILL.md:68: subject "문장은" with predicate "붙입니다" (a sentence does not attach particles).
    s = _line(SKILL, "틀에 번호를 끼워 넣는 문장은")
    subject = s >= 0 and "붙입니다." in SKILL[s]
    # SKILL.md:132 says "마쳤습니다" twice and adds "예제 코드", which the source line never says.
    src = _line(SKILL, "원문 (Claude 4277, DM-20)")
    rewrite = src >= 0 and SKILL[src + 1].count("마쳤습니다") == 2 \
        and "예제 코드" in SKILL[src + 1] and "예제" not in SKILL[src]
    print(f"SKILL.md lines: heading {h + 1}, subject {s + 1}, rewrite {src + 2}")
    return heading and subject and rewrite


def tests() -> bool:
    # Both particle tests (tests/test_polish_skill.py:101 and :186) only read the section-6 block,
    # so a doubled or mismatched particle in any other section's rewrite still passes.
    scoped = [i + 1 for i, l in enumerate(TEST) if '_block_with(_read_skill(), "을(를)", "(으)로", "받침")' in l]
    whole = any(re.search(r"\(\?:를\|을", l) and "_pairs(_read_skill())" in l for l in TEST)
    print(f"block-scoped particle checks at tests/test_polish_skill.py:{scoped}")
    return len(scoped) >= 2 and not whole


CHECKS = {"frontmatter": frontmatter, "particles": particles, "content": content, "tests": tests}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = CHECKS[name]() if name in CHECKS else False
    print(f"evidence {name or '<none>'}: {'confirmed' if ok else 'not found'}")
    sys.exit(0 if ok else 1)
