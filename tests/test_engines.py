"""e-means and o-means reimplementations reproduce the behaviour their papers report."""

import math

import numpy as np
import pytest

from dmeans import EMeans, OMeans, lloyd
from dmeans.emeans import recipe
from dmeans.geometry import mse, nearest


def mixture(n, d, modes, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(modes, d)).astype(np.float32)
    return c[rng.integers(0, modes, n)] + 0.6 * rng.normal(size=(n, d)).astype(np.float32)


@pytest.fixture(scope="module")
def problem():
    x = mixture(42_000, 32, 1024)
    return x[:40_000], x[40_000:]


def test_recipe_follows_scaling_laws():
    r = recipe(1_000_000, 16_384, 96, 3)
    raw = 0.035134 * 16_384**0.134749 * 1e6**0.876473 * 3**0.813344
    assert r["batch_size"] == 1 << round(math.log2(raw))
    alpha = 0.8244 * 16_384**-0.4617 * r["batch_size"] ** 0.9003 * 1e6**-0.4498 * 3**-0.1057
    assert r["alpha0"] == pytest.approx(min(max(alpha, 1e-4), 1.0))
    assert recipe(1000, 4, 8, 5)["batch_law"] == "quality"  # d <= 16 uses the quality law
    assert recipe(100, 64, 768, 5)["batch_size"] <= 50  # capped near n / 2


def test_emeans_beats_lloyd25_in_fewer_epochs(problem):
    xtr, xte = problem
    c_lloyd, _ = lloyd(xtr, 256, iters=25, seed=1)
    ref = mse(xte, c_lloyd)
    c5, _ = EMeans(256, nepochs=5, seed=1).fit(xtr)
    c2, _ = EMeans(256, nepochs=2, seed=1).fit(xtr)
    assert mse(xte, c5) < ref
    assert mse(xte, c2) < ref * 1.01


def test_emeans_revives_dead_clusters(problem):
    xtr, _ = problem
    init = np.repeat(xtr[:8], 32, axis=0)  # 256 centroids stacked on 8 points
    c, _ = EMeans(256, nepochs=3, seed=0).fit(xtr, init=init)
    used = np.unique(nearest(xtr, c)[0]).size
    assert used > 0.95 * 256


def test_omeans_cuts_distances_at_equal_error(problem):
    xtr, xte = problem
    e = EMeans(1024, nepochs=3, seed=2)
    ce, _ = e.fit(xtr)
    o = OMeans(1024, nepochs=3, seed=2, nlist=32, nprobe=4)
    co, _ = o.fit(xtr)
    assert o.distance_evals <= e.distance_evals / 3
    assert mse(xte, co) <= mse(xte, ce) * 1.02
    assert o.recipe["closing"] == "last-pass"


def test_omeans_short_runs_close_with_recency_mean(problem):
    xtr, xte = problem
    o = OMeans(256, nepochs=1, seed=3)
    c, _ = o.fit(xtr)
    assert o.recipe["closing"] == "recency"
    assert np.isfinite(mse(xte, c))


@pytest.mark.parametrize("nepochs", [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.5])
def test_fractional_budget_reads_no_more_rows_than_official_emeans(nepochs):
    from faiss.contrib.e_means.base import _epoch_rows  # the official E-means' rows per budget

    x = np.random.default_rng(0).normal(size=(500, 768)).astype(np.float32)
    e = EMeans(18, nepochs=nepochs, seed=0)
    e.fit(x)
    assert e.epochs_run == sum(_epoch_rows(len(x), nepochs)) / len(x)


@pytest.mark.parametrize("nepochs", [1, 2, 4])
def test_whole_budget_keeps_full_batches(nepochs):
    x = np.random.default_rng(0).normal(size=(500, 768)).astype(np.float32)
    e = EMeans(18, nepochs=nepochs, seed=0)
    e.fit(x)
    b = e.recipe["batch_size"]
    assert e.epochs_run == nepochs * (len(x) // b) * b / len(x) <= nepochs
