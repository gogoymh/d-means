"""The official E-means and O-means baselines in bench/: the CPU port of O-means' Triton kernels
must compute what the kernels compute, and the whole trainer must reach E-means' error."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faiss")  # faiss itself runs in bench.official's worker process

from bench import official  # noqa: E402
from bench.third_party.o_means import cpu_kernels as K  # noqa: E402


def _table(k=300, d=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    lay = K.dense_layout(d, k)
    C = torch.zeros((lay.kp, lay.dp))
    C[:k, :d] = torch.randn(k, d, generator=g)
    c16 = torch.zeros((lay.kp, lay.dp), dtype=torch.float16)
    cn = torch.full((lay.kp,), float("inf"))
    K.refresh_table(C, c16, cn, k, d)
    x16, _ = K.cast_pad_rows(torch.randn(500, d, generator=g), lay.dp)
    return lay, C, c16, cn, x16


def test_probing_every_cell_is_the_exact_search():
    lay, C, c16, cn, x16 = _table()
    k, G = lay.k, 16
    order, cells = K.build_cells(C, k, G, 4, generator=torch.Generator().manual_seed(1))
    K.permute_rows(order, k, [C, c16, cn])
    dense = K.assign_dense(x16, c16, cn, lay, None)
    full = K.two_level_assign(x16, c16, cn, cells, None, G)
    assert torch.equal(full, dense)
    part = K.two_level_assign(x16, c16, cn, cells, None, 4)
    assert 0.5 < (part == dense).float().mean() < 1  # 4 of 16 cells: approximate, mostly right


def test_split_children_are_checked():
    lay, C, c16, cn, x16 = _table(k=64)
    order, cells = K.build_cells(C, 64, 16, 4, generator=torch.Generator().manual_seed(2))
    K.permute_rows(order, 64, [C, c16, cn])
    first = K.two_level_assign(x16[:1], c16, cn, cells, None, 1)
    j = int(first[0])
    other = int(cells.off[(int(torch.searchsorted(cells.off, j, right=True)) % 16)])  # a centroid in another cell
    c16[other] = x16[0]
    cn[other] = x16[0].float().square().sum()
    assert int(K.two_level_assign(x16[:1], c16, cn, cells, None, 1)[0]) == j  # not probed
    cells.child[j] = other
    assert int(K.two_level_assign(x16[:1], c16, cn, cells, None, 1)[0]) == other  # reached as j's child


def test_accumulate_and_ema_update_match_the_emeans_formula():
    lay, C, c16, cn, x16 = _table(k=40, d=16)
    k, d = 40, 16
    labels = K.assign_dense(x16, c16, cn, lay, None)
    sums = torch.zeros((lay.kp, lay.dp), dtype=torch.int64)
    cnt = torch.zeros(lay.kp, dtype=torch.int32)
    K.accumulate(x16, labels, sums, cnt, d)
    exact = torch.zeros((k, d), dtype=torch.float64).index_add_(0, labels, x16[:, :d].double())
    assert torch.equal(sums[:k, :d].double() / 2**24, exact)  # fixed point is exact
    usage = torch.rand(lay.kp)
    u0, c0, n = usage[:k].clone(), C[:k, :d].clone(), cnt[:k].float()
    alpha = 0.3
    K.ema_update(C, c16, cn, usage, sums, cnt, k, d, alpha)
    mass = (1 - alpha) * u0 + alpha * n
    mean = exact.float() / n.clamp_min(1)[:, None]
    want = torch.where((n > 0)[:, None], c0 + (alpha * n / mass)[:, None] * (mean - c0), c0)
    assert torch.allclose(C[:k, :d], want, atol=1e-5)
    assert torch.allclose(usage[:k], torch.where(n > 0, mass, (1 - alpha) * u0))
    assert int(cnt.sum()) == 0 and int(sums.abs().sum()) == 0  # cleared for the next batch


def test_split_separates_equal_fp16_children():
    lay, C, c16, cn, _ = _table(k=8, d=4)
    usage = torch.ones(lay.kp)
    C[0, :4] = 0.0
    K.refresh_table(C, c16, cn, 8, 4)
    K.apply_splits(C, c16, cn, usage, torch.tensor([0]), torch.tensor([5]), torch.tensor(1), 4, 1e-4, 1)
    assert not torch.equal(c16[0, :4], c16[5, :4]) and usage[0] == usage[5] == 0.5


def _gauss(n=4000, d=24, seed=0):
    """One Gaussian: a smooth k-means landscape, so runs differ by the trainer, not by the local minimum a seed lands in."""
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def _mse(x, c):
    d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1)
    return float(d2.min(1).mean())


@pytest.mark.parametrize("k", [32, 512])  # 512: 16 cells, 16 probes of 16; small enough to stay fast
def test_omeans_reaches_emeans_error(k):
    x = _gauss()
    ce, pe = official.emeans(x, k, nepochs=3, seed=0)
    co, po = official.omeans(x, k, nepochs=3, seed=0)
    assert pe == po == 3.0
    assert _mse(x, co) <= 1.02 * _mse(x, ce)
