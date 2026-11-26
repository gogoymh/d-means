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


_PAIR_RE = re.compile(r"(?m)^-\s*(원문|고친 예)[^:\n]*:\s*\n?\s*`([^`]+)`")


def _pairs(text: str) -> list[tuple[str, str]]:
    """Pair each before example with the rewrite that follows it."""
    pairs, before = [], None
    for kind, body in _PAIR_RE.findall(text):
        if kind == "원문":
            before = body
        elif before is not None:
            pairs.append((before, body))
            before = None
    return pairs


def _block_with(text: str, *phrases: str) -> str:
    blocks = [b for b in _sections(text) if all(p in b for p in phrases) and "원문" in b and "고친 예" in b]
    assert blocks, f"missing pattern block with before/after examples for: {phrases}"
    return blocks[0]


def test_particle_doubling_and_mismatch_pattern():
    block = _block_with(_read_skill(), "을(를)", "(으)로", "받침")
    pairs = _pairs(block)
    assert len(pairs) >= 4
    assert any("을(를)" in b for b, _ in pairs) and any("(으)로" in b for b, _ in pairs)
    for _, after in pairs:
        assert not re.search(r"\((?:를|을|으|는|은|가|이)\)", after), after
    rewrites = " ".join(a for _, a in pairs)
    for expected in ("DM-27을", "DM-22를", "DM-30을", "DM-15는"):
        assert expected in rewrites, expected


def test_nominal_cliche_patterns():
    text = _read_skill()
    for phrase in ("진행해야", "해당하고", "및", "누락 시"):
        pairs = [(b, a) for b, a in _pairs(_block_with(text, phrase)) if phrase in b]
        assert pairs, f"no before example contains: {phrase}"
        for _, after in pairs:
            assert phrase not in after, f"rewrite still contains {phrase}: {after}"


def test_english_mixing_pattern():
    pairs = _pairs(_block_with(_read_skill(), "done이", "is done"))
    assert len(pairs) >= 2
    for _, after in pairs:
        assert not re.search(r"\bdone\b", after), after
        assert after.rstrip().endswith("니다.") or after.rstrip().endswith("므로"), after


_ID_RE = re.compile(r"DM-\d+|#\d+|[A-Za-z0-9_./-]+\.(?:png|jpg|md|py|json)")


def test_rewrites_keep_identifiers():
    pairs = _pairs(_read_skill())
    assert len(pairs) >= 15
    for before, after in pairs:
        missing = [t for t in set(_ID_RE.findall(before)) if t not in after]
        assert not missing, f"rewrite dropped {missing}: {after}"


def test_checklist_covers_new_patterns():
    items = "\n".join(re.findall(r"(?m)^-\s*\[\s\]\s+.+$", _read_skill()))
    for phrase in ("을(를)", "진행하다", "done", "피동"):
        assert phrase in items, phrase
