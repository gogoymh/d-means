"""Tests for the project Korean writing polish skill (.claude/skills/polish/SKILL.md)."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / ".claude" / "skills" / "polish" / "SKILL.md"


def _read_skill() -> str:
    assert SKILL_PATH.is_file(), f"missing skill file: {SKILL_PATH}"
    return SKILL_PATH.read_text(encoding="utf-8")


def _sections(text: str) -> list[str]:
    """Split markdown text on level-2/3 headings so each pattern block can be checked."""
    return re.split(r"(?m)^#{2,3}\s+", text)


def test_skill_file_and_frontmatter():
    text = _read_skill()
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert match is not None, "SKILL.md must begin with YAML frontmatter"
    frontmatter = match.group(1)
    assert re.search(r"(?m)^name:\s*polish\s*$", frontmatter), "frontmatter must set name: polish"
    desc = re.search(r"(?m)^description:\s*(.+)$", frontmatter)
    assert desc is not None and len(desc.group(1).strip()) >= 15, "frontmatter must describe when to use the skill"


def test_translationese_patterns_and_rewrites():
    text = _read_skill()
    blocks = _sections(text)
    required_patterns = [
        ("무생물 주어", "보여 준다"),
        ("범위가 잘린 티켓", "범위가 잘린"),
        ("가장 분명한 작은 일", "가장 분명한 작은 일"),
        ("배너 렌더", "배너 렌더"),
    ]
    for label, phrase in required_patterns:
        matched = [b for b in blocks if phrase in b and "원문" in b and "고친 예" in b]
        assert matched, f"missing pattern block with before/after example for: {label}"
        block = matched[0]
        rewrites = re.findall(r"(?m)^-\s*고친 예[^:]*:\s*(.+)$", block)
        assert rewrites, f"missing rewrite line for: {label}"
        for line in rewrites:
            assert phrase not in line, f"rewrite for {label} still contains the raw phrase: {line}"


def test_register_mixing_pattern_and_rewrite():
    text = _read_skill()
    blocks = _sections(text)
    matched = [
        b
        for b in blocks
        if "존댓말" in b and "반말" in b and "원문" in b and "고친 예" in b
    ]
    assert matched, "missing section covering mixed polite/plain register with examples"
    block = matched[0]
    assert "DM-22" in block and "DM-25" in block
    assert "맡았습니다." in block and "병합했습니다" in block
    rewrites = re.findall(r"(?m)고친 예\s*\d*:\s*\n?\s*`([^`]+)`", block)
    assert len(rewrites) >= 2, "expected unified-register rewrites for both DM-22 and DM-25"
    for rewritten in rewrites:
        assert "작은 일이다." not in rewritten
        assert "보여 준다." not in rewritten
        assert "빠졌다." not in rewritten
        sentences = [s.strip() for s in re.split(r"\.(?:\s+|$)", rewritten) if s.strip()]
        assert sentences and all(s.endswith("니다") for s in sentences), sentences


def test_principles_and_checklist():
    text = _read_skill()
    assert "원래 뜻" in text
    assert "바꾸지 않" in text
    checklist_items = re.findall(r"(?m)^-\s*\[\s\]\s+.+$", text)
    assert len(checklist_items) >= 3, f"expected at least 3 checklist items, found {len(checklist_items)}"
