"""The README banner: a letter D of clustered spheres, in light and dark variants."""

import importlib.util
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


def blobs(theme):
    """Every sphere in the PNG: (label image, centres (y, x), areas, dominant channel)."""
    img = load(theme)
    saturated = np.ptp(img, axis=2) >= 24  # the neutral backdrop and wordmark fall below this
    labels, count = ndimage.label(saturated)
    idx = np.arange(1, count + 1)
    areas = ndimage.sum(saturated, labels, idx)
    keep = idx[areas >= 15]
    centres = np.array(ndimage.center_of_mass(saturated, labels, keep))
    dominant = np.array([np.bincount(img.argmax(axis=2)[labels == k], minlength=3).argmax() for k in keep])
    return saturated, centres, areas[keep - 1], dominant


@pytest.mark.parametrize("theme", BANNERS)
def test_d_made_of_spheres(theme):
    saturated, centres, _, _ = blobs(theme)
    assert not saturated[:, banner.WORDMARK_AT[0]:].any()
    assert len(centres) >= 20, len(centres)
    (top, left), (bottom, right) = centres.min(axis=0), centres.max(axis=0)
    h, w = bottom - top, right - left
    assert 0.6 < w / h < 1.2, (w, h)  # upright letter proportions
    # stem: a column of spheres down the left edge covering most of the height
    stem = centres[np.abs(centres[:, 1] - np.median(centres[centres[:, 1] < left + 0.2 * w, 1])) < 10]
    assert np.ptp(stem[:, 0]) > 0.75 * h and len(stem) >= 8, stem
    # bowl: spheres on the right at mid height, and an empty counter inside the D
    assert (np.abs(centres[centres[:, 1] > right - 0.15 * w, 0] - (top + bottom) / 2) < 0.2 * h).any()
    y0, y1, x0, x1 = (int(v) for v in (top + 0.3 * h, bottom - 0.3 * h, left + 0.2 * w, right - 0.4 * w))
    assert not saturated[y0:y1, x0:x1].any()


@pytest.mark.parametrize("theme", BANNERS)
def test_corners_are_denser(theme):
    _, centres, areas, dominant = blobs(theme)
    (top, left), (bottom, right) = centres.min(axis=0), centres.max(axis=0)
    mid_stem = np.array([(top + bottom) / 2, left])

    def crowd(at):
        return (np.linalg.norm(centres - at, axis=1) < 55).sum()

    # amber (red), teal (green) and violet (blue): each group's largest sphere is its centroid
    corners = []
    for channel in range(3):
        group = np.flatnonzero(dominant == channel)
        assert len(group) >= 10, (channel, len(group))
        biggest = group[np.argmax(areas[group])]
        assert areas[biggest] >= 3 * np.median(areas[group])
        corners.append(centres[biggest])
        assert crowd(centres[biggest]) >= 3 * crowd(mid_stem), (channel, crowd(centres[biggest]), crowd(mid_stem))
    amber, teal, violet = corners
    h, w = bottom - top, right - left
    assert np.linalg.norm(amber - (top, left)) < 0.2 * h  # top-left corner
    assert np.linalg.norm(violet - (bottom, left)) < 0.2 * h  # bottom-left corner
    assert teal[1] > right - 0.2 * w  # tip of the bowl
    # every sphere belongs to its nearest centroid, as in k-means
    nearest = np.linalg.norm(centres[:, None] - np.array(corners), axis=-1).argmin(axis=1)
    assert (nearest == dominant).mean() >= 0.95


def test_readme_shows_hero_banner():
    head = (ROOT / "README.md").read_text().split("</h1>")[0]
    assert '<img src="assets/d-means-hero.jpg"' in head
    assert (ASSETS / "d-means-hero.jpg").is_file()


def test_generated_banner_theme_dimensions_and_brightness():
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
