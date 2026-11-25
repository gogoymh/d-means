"""The README banner: three smooth spheres, light and dark variants, rebuilt by assets/banner.py."""

import importlib.util
import itertools
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BANNERS = {"light": ASSETS / "d-means-banner.png", "dark": ASSETS / "d-means-banner-dark.png"}
OLD_BYTES = 757_387  # the beaded banner this one replaced (commit f6e67ff)

_spec = importlib.util.spec_from_file_location("banner", ASSETS / "banner.py")
banner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(banner)


def load(theme):
    return banner.png_read(BANNERS[theme])[..., :3]


def luma(img):
    return img.astype(float) @ [0.2126, 0.7152, 0.0722]


def bead_metrics(img):
    """Strong-edge pixels and bright highlight peaks in the object half of the banner.

    The beaded banner (f6e67ff) scores 22,143 edge pixels and 69 peaks: every bead adds an
    outline and a highlight. Three smooth spheres score about 5,000 and 13.
    """
    g = luma(img)[:, :700]
    edges = int((np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0)) > 60).sum())
    blur = ndimage.gaussian_filter(g, 2)
    peaks = (blur == ndimage.maximum_filter(blur, 15)) & (blur - ndimage.minimum_filter(blur, 31) > 25)
    return edges, ndimage.label(peaks)[1]


def test_scene_is_three_large_spheres_apart():
    centers, radii = banner.spheres()
    assert len(radii) == 3
    assert radii.min() >= 0.7 * radii.max()  # no small beads
    assert np.allclose(centers[:, 1], radii)  # each rests on the floor
    for i, j in itertools.combinations(range(3), 2):
        assert np.linalg.norm(centers[i] - centers[j]) > radii[i] + radii[j]  # none sink into another


@pytest.mark.parametrize("theme", BANNERS)
def test_banner_has_no_beads(theme):
    edges, peaks = bead_metrics(load(theme))
    assert edges <= 8_000, edges
    assert peaks <= 20, peaks


def test_readme_shows_dark_and_light_banners():
    head = (ROOT / "README.md").read_text().split("</h1>")[0]
    assert re.search(r'<source media="\(prefers-color-scheme: dark\)" srcset="assets/d-means-banner-dark.png">', head)
    assert '<img src="assets/d-means-banner.png"' in head
    light, dark = luma(load("light")) / 255, luma(load("dark")) / 255
    assert light.shape == dark.shape == (512, 1536)
    assert light.mean() > 0.75 and dark.mean() < 0.2, (light.mean(), dark.mean())


def contrast(a, b):
    def lum(c):
        c = np.asarray(c) / 255
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return c @ [0.2126, 0.7152, 0.0722]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", BANNERS)
def test_wordmark_contrast(theme):
    mask = banner.png_read(ASSETS / "wordmark.png")[..., 0]
    x, y = banner.WORDMARK_AT
    box = load(theme)[y: y + mask.shape[0], x: x + mask.shape[1]].astype(float)
    ink, ground = box[mask == 255].mean(0), box[mask == 0].mean(0)
    assert (mask == 255).sum() > 10_000  # the whole word is there, not a fragment
    assert contrast(ink, ground) >= 7.0, contrast(ink, ground)


def test_banners_are_smaller_than_the_old_one():
    sizes = [p.stat().st_size for p in BANNERS.values()]
    assert sum(sizes) < OLD_BYTES, sizes


@pytest.mark.parametrize("theme", BANNERS)
def test_generator_reproduces_committed_banner(theme):
    fresh = banner.render(theme).astype(int)
    diff = np.abs(fresh - load(theme).astype(int))
    assert diff.max() <= 2 and diff.mean() < 0.05, (diff.max(), diff.mean())
