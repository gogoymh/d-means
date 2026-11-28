"""README figures: every asset generator writes only committed files, and none leaves stray previews behind."""

import subprocess
from pathlib import Path

from assets import steps

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()


def test_banner_generator_is_gone():
    # DM-22's banner.py --preview left an untracked PNG in assets/; the generator was removed, so the README must not point at it.
    assert not (ROOT / "assets" / "banner.py").exists()
    readme = (ROOT / "README.md").read_text()
    assert "banner.py" not in readme and "--preview" not in readme


def test_steps_writes_only_tracked_files(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "HERE", tmp_path)
    steps.main()
    written = sorted(p.name for p in tmp_path.iterdir())
    tracked = {Path(p).name for p in _git("ls-files", "assets")}
    assert written and set(written) <= tracked, set(written) - tracked
    for name in written:  # regenerating reproduces the committed figure byte for byte
        assert (tmp_path / name).read_text() == (ROOT / "assets" / name).read_text(), name


def test_assets_folder_has_no_untracked_files():
    assert _git("ls-files", "--others", "--exclude-standard", "assets") == []
