"""Distance kernels shared by every algorithm (float32, BLAS matmul)."""

import numpy as np


def sqnorms(x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", x, x)


def sqdist(x: np.ndarray, c: np.ndarray, xn: np.ndarray | None = None, cn: np.ndarray | None = None) -> np.ndarray:
    """Squared L2 distances, shape (len(x), len(c)), clipped at 0."""
    xn = sqnorms(x) if xn is None else xn
    cn = sqnorms(c) if cn is None else cn
    d = x @ c.T  # in place, same rounding as xn - 2 x.c + cn
    d *= -2.0
    d += xn[:, None]
    d += cn[None, :]
    np.maximum(d, 0.0, out=d)
    return d


def nearest(x: np.ndarray, c: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Index and squared distance of each row's nearest centroid."""
    cn = sqnorms(c)
    idx = np.empty(len(x), dtype=np.int64)
    dist = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), chunk):
        d = sqdist(x[s : s + chunk], c, cn=cn)
        j = d.argmin(1)
        idx[s : s + chunk] = j
        dist[s : s + chunk] = d[np.arange(len(j)), j]
    return idx, dist


def mse(x: np.ndarray, c: np.ndarray) -> float:
    """Mean squared distance of rows to their nearest centroid (float64 sum)."""
    return float(nearest(x, c)[1].astype(np.float64).mean())


def cluster_sums(x: np.ndarray, labels: np.ndarray, k: int, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster (weighted) counts and coordinate sums."""
    w = np.ones(len(labels), dtype=np.float64) if weights is None else weights.astype(np.float64)
    counts = np.bincount(labels, weights=w, minlength=k)
    order = np.argsort(labels, kind="stable")
    ls = labels[order]
    starts = np.searchsorted(ls, np.arange(k))
    xw = x[order] * w[order, None].astype(x.dtype)
    sums = np.zeros((k, x.shape[1]), dtype=np.float64)
    nz = np.flatnonzero(np.bincount(labels, minlength=k))
    if nz.size:
        red = np.add.reduceat(xw, starts[nz], axis=0)
        sums[nz] = red
    return counts, sums
