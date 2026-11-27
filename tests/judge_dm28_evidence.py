"""Evidence checks for the DM-31 grading of DM-28 (.claude/skills/polish/SKILL.md).

Not collected by pytest on purpose: each check confirms that a defect cited in the
follow-up ticket still exists at the cited file:line. Run one check by name:

    uv run python tests/judge_dm28_evidence.py frontmatter|register|label|test
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / ".claude/skills/polish/SKILL.md").read_text(encoding="utf-8").splitlines()
TEST = (ROOT / "tests/test_polish_skill.py").read_text(encoding="utf-8").splitlines()


def frontmatter() -> bool:
    # YAML plain scalars end at " #": the description is cut to "d-means 프로젝트의".
    line = SKILL[2]
    value = line.split(":", 1)[1].strip()
    return line.startswith("description:") and not value.startswith(("'", '"')) and " #" in value


def register() -> bool:
    # Line 15 says criteria use plain style, yet lines 23-26 rewrite DM-25/DM-26 criteria into polite style.
    return "완료 기준" in SKILL[14] and "평서체" in SKILL[14] and SKILL[23].rstrip("`").endswith("나옵니다.") \
        and SKILL[25].rstrip("`").endswith("나옵니다.")


def label() -> bool:
    # Heading on line 17 scopes the section to DM-22 and DM-25, but line 25 cites DM-26.
    return "DM-22와 DM-25" in SKILL[16] and "(DM-26)" in SKILL[24]


def test() -> bool:
    # Line 26 checks the description with a regex, so the YAML truncation passes unnoticed.
    return "re.search" in TEST[25] and "description" in TEST[25] and "yaml" not in "\n".join(TEST)


CHECKS = {"frontmatter": frontmatter, "register": register, "label": label, "test": test}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    ok = CHECKS[name]() if name in CHECKS else False
    print(f"evidence {name or '<none>'}: {'confirmed' if ok else 'not found'}")
    sys.exit(0 if ok else 1)
