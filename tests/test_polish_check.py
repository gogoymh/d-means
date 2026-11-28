"""Tests for tools/polish_check.py against real #general messages."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "polish_check.py"
FIXTURE = ROOT / "tests" / "fixtures" / "general_messages.txt"

_spec = importlib.util.spec_from_file_location("polish_check", SCRIPT)
pc = importlib.util.module_from_spec(_spec)
sys.modules["polish_check"] = pc
_spec.loader.exec_module(pc)


def _messages() -> dict[str, str]:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return dict(pc.split_header(line, "?") for line in lines if line.strip())


def _rules(text: str) -> set[str]:
    return {v.rule for v in pc.check(text)}


def test_particle_doubling_in_general_with_fitting_particle():
    msgs = _messages()
    expected = {"#11664": "DM-27을", "#11933": "DM-31을", "#12313": "DM-30을",
                "#13723": "DM-34를", "#14600": "DM-32를", "#22025": "DM-40을"}
    for mid, fix in expected.items():
        found = [v for v in pc.check(msgs[mid]) if v.rule == "particle"]
        assert len(found) == 1, (mid, found)
        assert found[0].hint == f"'{fix}'", (mid, found[0])
        assert msgs[mid][found[0].col - 1:].startswith(found[0].text)


def test_particle_follows_final_consonant():
    hints = [v.hint for v in pc.check("서울(으)로 갑니다. 파일(으)로 둡니다. 집(으)로 갑니다. DM-17은(는) 끝났습니다. 차와(과) 집과(와) 갑니다.")]
    assert hints == ["'서울로'", "'파일로'", "'집으로'", "'DM-17은'", "'차와'", "'집과'"], hints


def test_english_status_words_in_general():
    msgs = _messages()
    for mid, word in {"#13052": "is done", "#11933": "done", "#13723": "done"}.items():
        found = [v for v in pc.check(msgs[mid]) if v.rule == "english"]
        assert [v.text for v in found] == [word], (mid, found)


def test_register_mixing_flags_plain_ending():
    text = "미션 DM-22를 맡았습니다. 코드만으로 끝낼 수 있는 배너 미리보기가 지금 가장 분명한 작은 일이다."
    found = [v for v in pc.check(text) if v.rule == "register"]
    assert [v.text for v in found] == ["일이다"]
    assert text[found[0].col - 1:].startswith("일이다")
    assert _rules("정렬한다. 테스트를 돌린다.") == set()  # one register throughout is fine


def test_clean_and_quoted_messages_pass():
    msgs = _messages()
    for mid in ("#14260", "#15699", "#22204"):
        assert pc.check(msgs[mid]) == [], mid
    # '#12313' quotes judge's '을(를)' on purpose; only its own '(으)로' counts.
    assert [v.text for v in pc.check(msgs["#12313"])] == ["(으)로"]
    assert pc.check("`done` 상태와 '작은 일이다.' 같은 예를 인용했습니다.") == []
    assert pc.check("The job is done.") == []  # English-only messages are out of scope


def test_cli_prints_locations_and_exit_status():
    run = subprocess.run([sys.executable, str(SCRIPT)], input=FIXTURE.read_text(encoding="utf-8"),
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 1
    lines = run.stdout.splitlines()
    assert "#11664:9: particle '을(를)' -> 'DM-27을'" in lines
    assert any(l.startswith("#13052:42: english 'is done'") for l in lines)
    assert not any(l.startswith(("#14260:", "#15699:", "#22204:")) for l in lines)
    ok = subprocess.run([sys.executable, str(SCRIPT), "DM-26을 마쳤습니다."], capture_output=True, text=True, timeout=30)
    assert ok.returncode == 0 and ok.stdout == ""
