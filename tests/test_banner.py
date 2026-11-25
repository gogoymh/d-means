"""The README banner: three clusters in light and dark variants."""

import importlib.util
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


@pytest.mark.parametrize("theme", BANNERS)
def test_scene_shows_three_separated_clusters(theme):
    img = load(theme)
    saturated = np.ptp(img, axis=2) >= 24
    assert not saturated[:, banner.WORDMARK_AT[0]:].any()
    dominant = img.argmax(axis=2)
    centroids, members = [], []

    # The neutral studio backdrop falls below the saturation threshold. Amber, teal and
    # violet have different dominant channels, so each sphere is measurable in the PNG.
    for channel in range(3):
        labels, count = ndimage.label(saturated & (dominant == channel))
        areas = np.bincount(labels.ravel())[1:]
        visible = np.flatnonzero(areas >= 40) + 1
        assert len(visible) == 6, (theme, channel, count, areas)
        order = visible[np.argsort(areas[visible - 1])[::-1]]
        assert areas[order[0] - 1] >= 4 * areas[order[1] - 1]
        center, *points = ndimage.center_of_mass(saturated, labels, order)
        points = np.array(points)
        assert np.all(points.min(axis=0) < center)
        assert np.all(points.max(axis=0) > center)
        centroids.append(center)
        members.append(points)

    distances = np.linalg.norm(np.array(members)[:, :, None] - np.array(centroids), axis=-1)
    assert np.array_equal(distances.argmin(axis=-1), np.repeat(np.arange(3)[:, None], 5, axis=1))


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
