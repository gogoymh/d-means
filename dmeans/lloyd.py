"""Lloyd's full-batch k-means, the reference every other method is measured against."""

from collections.abc import Callable

import numpy as np

from . import _log
from .geometry import cluster_sums, nearest

log = _log.get("lloyd")


def init_centroids(x: np.ndarray, k: int, seed: int, method: str = "random") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "random":
        return x[rng.choice(len(x), k, replace=False)].astype(np.float32, copy=True)
    if method == "k-means++":
        from sklearn.cluster import kmeans_plusplus

        c, _ = kmeans_plusplus(x, k, random_state=seed)
        return c.astype(np.float32)
    raise ValueError(f"unknown init {method!r}")


def lloyd(
    x: np.ndarray,
    k: int,
    iters: int = 25,
    seed: int = 1234,
    init: str | np.ndarray = "random",
    on_epoch: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `iters` Lloyd iterations (one epoch each). Empty clusters keep their centroid."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    c = init if isinstance(init, np.ndarray) else init_centroids(x, k, seed, init)
    c = c.astype(np.float32, copy=True)
    labels = np.zeros(len(x), dtype=np.int64)
    for it in range(iters):
        labels, _ = nearest(x, c)
        counts, sums = cluster_sums(x, labels, k)
        live = counts > 0
        c[live] = (sums[live] / counts[live, None]).astype(np.float32)
        if on_epoch:
            on_epoch(it + 1, c)
    log.debug("lloyd done n=%d k=%d iters=%d empty=%d", len(x), k, iters, int((~live).sum()))
    return c, labels
