"""README hero: the header image is the owner's sculpted-D JPEG, referenced with the right attributes.

Checks parse the HTML attributes and the image file instead of pinning README prose.
"""

import hashlib
import struct
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
HERO = "assets/d-means-hero.jpg"
# Git blob id of the hero from 2494a52/d4f3336/dce1047 (as at 151451b). Any edit must be deliberate.
HERO_BLOB = "c6e91b6c331e02a7dd3c0101711c0759dc1a453f"


class _Images(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str | None, dict[str, str]]] = []
        self._stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append((self._stack[-1] if self._stack else None, {k: v or "" for k, v in attrs}))
        else:
            self._stack.append(tag)

    def handle_endtag(self, tag):
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()


def _images() -> list[tuple[str | None, dict[str, str]]]:
    parser = _Images()
    parser.feed(README.read_text(encoding="utf-8"))
    return parser.images


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the first SOF marker of a baseline or progressive JPEG."""
    assert data[:2] == b"\xff\xd8", "not a JPEG (missing SOI)"
    i = 2
    while i < len(data):
        assert data[i] == 0xFF, f"bad marker at byte {i}"
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + length
    raise AssertionError("no SOF marker")


def test_hero_is_first_image_inside_h1():
    parent, attrs = _images()[0]
    assert parent == "h1"
    assert attrs["src"] == HERO
    assert attrs.get("width") == "960"


def test_hero_alt_names_project_and_describes_the_image():
    alt = _images()[0][1].get("alt", "")
    assert alt.startswith("D-means")
    # The alt must describe what the JPEG shows (a sculpted D with sphere clusters), not a beaded banner.
    assert "sculpted D" in alt and "spheres" in alt
    assert "bead" not in alt.lower()


def test_hero_file_is_the_owner_jpeg():
    data = (ROOT / HERO).read_bytes()
    assert data.endswith(b"\xff\xd9"), "JPEG truncated (missing EOI)"
    width, height = _jpeg_size(data)
    assert (width, height) == (1600, 680)
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == HERO_BLOB


def test_every_readme_image_exists():
    readme = README.read_text(encoding="utf-8")
    srcs = [a["src"] for _, a in _images() if "src" in a]
    srcs += [part.split('"')[0] for part in readme.split('srcset="')[1:]]
    missing = [s for s in srcs if not (ROOT / s).is_file()]
    assert not missing, missing


def test_no_stale_banner_assets():
    assert not (ROOT / "assets" / "banner.py").exists()
    assert not list((ROOT / "assets").glob("d-means-banner*"))
