"""e-means (Bessa et al., arXiv:2611.09501), reimplemented from the paper for NumPy on CPU.

Mini-batch k-means whose centroids are exponential moving averages of the points assigned
to them. The batch size and peak step come from the paper's scaling laws, the step follows
a cosine schedule, dead clusters are revived by splitting a donor every five updates, and the
run ends with the last-pass mean.
"""

import math
from collections.abc import Callable

import numpy as np

from . import _log
from .geometry import cluster_sums, sqdist, sqnorms

log = _log.get("emeans")

TAU = 0.1  # revival threshold, fraction of the mean count
EPS = 0.1  # cosine floor of the step schedule
DELTA = 2.0**-7  # split perturbation (BF16 machine epsilon)
REVIVE_EVERY = 5


def _pow2(v: float) -> int:
    return 1 << max(0, int(round(math.log2(max(v, 1.0)))))


def recipe(n: int, k: int, d: int, nepochs: float, batch_law: str = "auto") -> dict:
    """Batch size and peak step from the e-means scaling laws (paper Table 2, README eqs. 1-3)."""
    t = max(float(nepochs), 1.0)
    law = ("quality" if d <= 16 else "throughput") if batch_law == "auto" else batch_law
    coef = {"throughput": 0.035134, "quality": 0.00439175}[law]
    b = _pow2(coef * k**0.134749 * n**0.876473 * t**0.813344)
    cap = 1 << max(0, int(math.floor(math.log2(max(n // 2, 1)))))
    b = max(1, min(b, cap))
    alpha0 = 0.8244 * k**-0.4617 * b**0.9003 * n**-0.4498 * t**-0.1057
    alpha0 = float(min(max(alpha0, 1e-4), 1.0))
    steps_per_pass = max(1, n // b)
    return {
        "batch_law": law,
        "batch_size": b,
        "alpha0": alpha0,
        "steps_per_pass": steps_per_pass,
        "total_steps": max(1, int(round(nepochs * steps_per_pass))),
        "revival": d > 2,
    }


def cosine_step(alpha0: float, t: int, total: int) -> float:
    return alpha0 * (EPS + (1.0 - EPS) * 0.5 * (1.0 + math.cos(math.pi * t / total)))


class EMeans:
    """e-means trainer. `fit(x)` returns (centroids, labels of the last pass)."""

    name = "e-means"

    def __init__(self, k: int, nepochs: float = 5, seed: int = 1234, batch_law: str = "auto", batch_size: int | None = None):
        self.k, self.nepochs, self.seed, self.batch_law, self.batch_size = k, nepochs, seed, batch_law, batch_size
        self.distance_evals = 0

    # --- assignment (o-means overrides this) -------------------------------------------
    def _assign(self, xb: np.ndarray, xbn: np.ndarray, c: np.ndarray) -> np.ndarray:
        self.distance_evals += len(xb) * len(c)
        return sqdist(xb, c, xn=xbn).argmin(1)

    def _on_split(self, donors: np.ndarray, children: np.ndarray, c: np.ndarray) -> None:
        pass

    def _after_step(self, t: int, rec: dict, c: np.ndarray, revived: bool) -> None:
        pass

    def _begin(self, x: np.ndarray, c: np.ndarray, rec: dict) -> None:
        pass

    # --- training ------------------------------------------------------------------------
    def _revive(self, rng: np.random.Generator, n_mass: np.ndarray, s: np.ndarray) -> int:
        dead = np.flatnonzero(n_mass < TAU * n_mass.mean())
        if dead.size == 0:
            return 0
        p = n_mass.copy()
        p[dead] = 0.0
        live = np.count_nonzero(p)
        dead = dead[: min(dead.size, live)]
        if dead.size == 0:
            return 0
        donors = rng.choice(self.k, size=dead.size, replace=False, p=p / p.sum())
        c_don = s[donors] / n_mass[donors, None]
        sign = np.where(np.arange(s.shape[1]) % 2 == 0, 1.0, -1.0)
        half = n_mass[donors] / 2.0
        n_mass[donors] = half
        n_mass[dead] = half
        s[dead] = c_don * (1.0 + DELTA * sign) * half[:, None]
        s[donors] = c_don * (1.0 - DELTA * sign) * half[:, None]
        self._on_split(donors, dead, s / np.maximum(n_mass, 1e-30)[:, None])
        return int(dead.size)

    def fit(
        self,
        x: np.ndarray,
        init: np.ndarray | None = None,
        on_step: Callable[[int, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        n, d = x.shape
        k = self.k
        rng = np.random.default_rng(self.seed)
        rec = recipe(n, k, d, self.nepochs, self.batch_law)
        if self.batch_size:
            rec["batch_size"] = self.batch_size
            rec["steps_per_pass"] = max(1, n // self.batch_size)
            rec["total_steps"] = max(1, int(round(self.nepochs * rec["steps_per_pass"])))
        self.recipe = rec
        b, total = rec["batch_size"], rec["total_steps"]
        c0 = x[rng.choice(n, k, replace=False)] if init is None else np.asarray(init, dtype=np.float32)
        n_mass = np.full(k, 1e-10)
        s = c0.astype(np.float64) * n_mass[:, None]
        c = c0.astype(np.float32, copy=True)
        xn = sqnorms(x)
        labels = np.full(n, -1, dtype=np.int64)
        self.distance_evals = 0
        self._begin(x, c, rec)
        t, revived_total = 0, 0
        while t < total:
            perm = rng.permutation(n)
            for p in range(rec["steps_per_pass"]):
                if t >= total:
                    break
                rows = perm[p * b : (p + 1) * b]
                xb = x[rows]
                a = self._assign(xb, xn[rows], c)
                labels[rows] = a
                cnt, sm = cluster_sums(xb, a, k)
                alpha = cosine_step(rec["alpha0"], t, total)
                n_mass *= 1.0 - alpha
                n_mass += alpha * cnt
                s *= 1.0 - alpha
                s += alpha * sm
                self._accumulate(t, rows, a, xb, cnt, sm)
                t += 1
                revived = False
                if rec["revival"] and t % REVIVE_EVERY == 0:
                    nr = self._revive(rng, n_mass, s)
                    revived_total += nr
                    revived = nr > 0
                c = (s / n_mass[:, None]).astype(np.float32)
                self._after_step(t, rec, c, revived)
                if on_step:
                    on_step(t, c)
        c = self._close(x, labels, c, rec)
        self.centroids, self.labels = c, labels
        log.debug("%s done n=%d k=%d steps=%d batch=%d alpha0=%.4f revived=%d", self.name, n, k, total, b, rec["alpha0"], revived_total)
        return c, labels

    def _accumulate(self, t, rows, a, xb, cnt, sm) -> None:
        pass

    def _close(self, x: np.ndarray, labels: np.ndarray, c: np.ndarray, rec: dict) -> np.ndarray:
        """Last-pass mean: each centroid becomes the mean of the rows last assigned to it."""
        seen = labels >= 0
        counts, sums = cluster_sums(x[seen], labels[seen], self.k)
        live = counts > 0
        out = c.copy()
        out[live] = (sums[live] / counts[live, None]).astype(np.float32)
        return out
