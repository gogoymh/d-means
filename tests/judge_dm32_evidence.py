"""Evidence checks for the DM-35 grading of DM-32 (.claude/skills/polish/SKILL.md).

Not collected by pytest on purpose: each check confirms a finding cited in the
grading at the cited file:line. Run one check by name:

    uv run python tests/judge_dm32_evidence.py description|checklist|imperative|tests

description confirms the defect DM-32 fixed; the others confirm defects still open.
"""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL_FILE = ".claude/skills/polish/SKILL.md"
SKILL = (ROOT / SKILL_FILE).read_text(encoding="utf-8").splitlines()
TEST = (ROOT / "tests/test_polish_skill.py").read_text(encoding="utf-8").splitlines()


def _yaml_description(text: str) -> str:
    """Parse the frontmatter description with a minimal YAML scalar reader (no dependency)."""
    fm = text.split("---\n")[1]
    raw = re.search(r"(?m)^description:[ \t]*(.*)$", fm).group(1).strip()
    if raw.startswith('"'):
        end = re.search(r'(?<!\\)"', raw[1:])
        return raw[1:1 + end.start()] if end else ""
    # A plain scalar ends at the first " #" (YAML comment).
    return raw.split(" #", 1)[0].rstrip()


def description() -> bool:
    # SKILL.md:3 before DM-32 (17c833b) was cut at " #general"; now it reads whole.
    before = subprocess.run(["git", "show", f"17c833b:{SKILL_FILE}"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    old, new = _yaml_description(before), _yaml_description("\n".join(SKILL) + "\n")
    print(f"before DM-32: {old!r}\nafter DM-32 ends with: {new[-12:]!r}")
    return old == "d-means 프로젝트의" and "#general" in new and new.endswith("가이드입니다.")


def checklist() -> bool:
    # SKILL.md:15 says every list item is written in 합쇼체, yet the checklist (SKILL.md:143-151)
    # ends each item in the 해라체 question "~는가?" or "~은가?", including line 145 that DM-32 rewrote.
    rule = next(i for i, l in enumerate(SKILL) if "종결어미 통일" in l)
    says_lists = "항목 목록" in SKILL[rule] and "합쇼체" in SKILL[rule]
    items = [(i + 1, l) for i, l in enumerate(SKILL) if re.match(r"^-\s*\[\s\]", l)]
    plain = [n for n, l in items if re.search(r"[는은]가\?$", l.rstrip())]
    print(f"rule SKILL.md:{rule + 1}; 해라체 checklist lines: {plain}")
    return says_lists and len(plain) == len(items) >= 9 and 145 in plain


def imperative() -> bool:
    # tests/test_polish_skill.py:172 accepts only "~니다." so a 합쇼체 imperative ("~십시오.")
    # fails; SKILL.md:93 turned the instruction "넣으세요" into the statement "넣습니다".
    t = next((i for i, l in enumerate(TEST) if 's.endswith("니다.")' in l), -1)
    s = next((i for i, l in enumerate(SKILL) if "21을(를) 넣으세요" in l), -1)
    rewrite = SKILL[s + 1] if s >= 0 else ""
    print(f"test line {t + 1}; rewrite SKILL.md:{s + 2}")
    return t >= 0 and "넣습니다." in rewrite and "십시오" not in rewrite


def tests() -> bool:
    # :126 still lets a rewrite end in "므로" though :172 forbids it; :182 pins exactly one issue
    # title, so adding a second title example breaks it; :187 leaves out the DM-32 case SKILL.md:90 adds.
    stale = any('endswith("므로")' in l for l in TEST)
    pinned = any("len(titles) == 1" in l for l in TEST)
    expected = next((l for l in TEST if l.strip().startswith("expected = {")), "")
    has_case = any("(polish, DM-32)" in l for l in SKILL)
    print(f"stale 므로: {stale}, pinned title count: {pinned}, DM-32 case untested: {has_case and 'DM-32' not in expected}")
    return stale and pinned and has_case and "DM-32" not in expected


CHECKS = {"description": description, "checklist": checklist, "imperative": imperative, "tests": tests}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = CHECKS[name]() if name in CHECKS else False
    print(f"evidence {name or '<none>'}: {'confirmed' if ok else 'not found'}")
    sys.exit(0 if ok else 1)
