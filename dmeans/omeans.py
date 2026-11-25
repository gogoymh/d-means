"""o-means (faiss contrib/o_means), reimplemented from its README for NumPy on CPU.

e-means with a two-level assignment: the k centroids are grouped into `nlist` coarse cells
by Lloyd iterations on the centroids, and each row is compared only with the centroids of
its `nprobe` nearest cells (plus the child and grandchild split from its best match since the
last rebuild). Cells are re-sorted after every revival check in the first pass and rebuilt
every half pass after it. Runs shorter than 2 epochs close with a recency-weighted mean.
"""

import math

import numpy as np

from . import _log
from .emeans import EMeans
from .geometry import cluster_sums, nearest, sqdist, sqnorms

log = _log.get("omeans")


def default_cells(k: int, d: int) -> tuple[int, int]:
    nlist = max(16, int(math.ceil((k // 128) / 16.0)) * 16)
    nlist = min(nlist, k)
    nprobe = 24 if k > 16384 else (12 if d >= 512 else 16)
    return nlist, min(nprobe, nlist)


class OMeans(EMeans):
    name = "o-means"

    def __init__(self, k: int, nepochs: float = 5, seed: int = 1234, nlist: int | None = None, nprobe: int | None = None, **kw):
        super().__init__(k, nepochs=nepochs, seed=seed, **kw)
        self._nlist, self._nprobe = nlist, nprobe

    # --- cells ---------------------------------------------------------------------------
    def _set_cells(self, cell_of: np.ndarray) -> None:
        self.cell_of = cell_of
        order = np.argsort(cell_of, kind="stable")
        bounds = np.searchsorted(cell_of[order], np.arange(self.nlist + 1))
        self.cells = [order[bounds[i] : bounds[i + 1]] for i in range(self.nlist)]
        self.link[:] = -1

    def _lloyd_cells(self, c: np.ndarray, iters: int) -> None:
        for _ in range(iters):
            cell_of, _ = nearest(c, self.cc)
            cnt, sm = cluster_sums(c, cell_of, self.nlist)
            live = cnt > 0
            self.cc[live] = (sm[live] / cnt[live, None]).astype(np.float32)
        self._set_cells(nearest(c, self.cc)[0])

    def _begin(self, x: np.ndarray, c: np.ndarray, rec: dict) -> None:
        n, d = x.shape
        nl, npb = default_cells(self.k, d)
        self.nlist = self._nlist or nl
        self.nprobe = min(self._nprobe or npb, self.nlist)
        rng = np.random.default_rng(self.seed + 1)
        self.cc = c[rng.choice(self.k, self.nlist, replace=False)].copy()
        self.link = np.full(self.k, -1, dtype=np.int64)
        self._lloyd_cells(c, 10)
        self.recency = rec["total_steps"] < 2 * rec["steps_per_pass"]
        if self.recency:
            self._w = np.zeros(self.k)
            self._z = np.zeros((self.k, d))
            self._rows_seen = 0
            self._n = n
        rec.update(nlist=self.nlist, nprobe=self.nprobe, closing="recency" if self.recency else "last-pass")

    def _after_step(self, t: int, rec: dict, c: np.ndarray, revived: bool) -> None:
        spp = rec["steps_per_pass"]
        if t <= spp:
            if t % 5 == 0:
                self._set_cells(nearest(c, self.cc)[0])
            if t == spp:
                self._lloyd_cells(c, 2)
            return
        half = max(1, spp // 2)
        if (t - spp) % half == 0:
            self._lloyd_cells(c, 2)

    def _on_split(self, donors: np.ndarray, children: np.ndarray, c: np.ndarray) -> None:
        self.link[donors] = children
        self.link[children] = -1
        if self.recency:
            self._w[donors] /= 2.0
            self._z[donors] /= 2.0
            self._w[children] = self._w[donors]
            self._z[children] = self._z[donors]

    # --- assignment ----------------------------------------------------------------------
    def _assign(self, xb: np.ndarray, xbn: np.ndarray, c: np.ndarray) -> np.ndarray:
        bsz = len(xb)
        cn = sqnorms(c)
        dcell = sqdist(xb, self.cc, xn=xbn)
        self.distance_evals += bsz * self.nlist
        if self.nprobe >= self.nlist:
            probe_mask = np.ones((bsz, self.nlist), dtype=bool)
        else:
            probe = np.argpartition(dcell, self.nprobe - 1, axis=1)[:, : self.nprobe]
            probe_mask = np.zeros((bsz, self.nlist), dtype=bool)
            probe_mask[np.arange(bsz)[:, None], probe] = True
        best = np.full(bsz, np.inf, dtype=np.float32)
        arg = np.full(bsz, -1, dtype=np.int64)
        for cell, members in enumerate(self.cells):
            if members.size == 0:
                continue
            rows = np.flatnonzero(probe_mask[:, cell])
            if rows.size == 0:
                continue
            dd = sqdist(xb[rows], c[members], xn=xbn[rows], cn=cn[members])
            self.distance_evals += rows.size * members.size
            j = dd.argmin(1)
            v = dd[np.arange(rows.size), j]
            better = v < best[rows]
            best[rows[better]] = v[better]
            arg[rows[better]] = members[j[better]]
        miss = arg < 0
        if miss.any():
            dd = sqdist(xb[miss], c, xn=xbn[miss], cn=cn)
            self.distance_evals += int(miss.sum()) * self.k
            arg[miss] = dd.argmin(1)
            best[miss] = dd.min(1)
        hop = self.link[arg]
        for _ in range(2):
            has = np.flatnonzero(hop >= 0)
            if has.size == 0:
                break
            cand = hop[has]
            v = np.maximum(xbn[has] - 2.0 * np.einsum("ij,ij->i", xb[has], c[cand]) + cn[cand], 0.0)
            self.distance_evals += has.size
            better = v < best[has]
            best[has[better]] = v[better]
            arg[has[better]] = cand[better]
            hop = np.full(bsz, -1, dtype=np.int64)
            hop[has] = self.link[cand]
        return arg

    # --- closing -------------------------------------------------------------------------
    def _accumulate(self, t, rows, a, xb, cnt, sm) -> None:
        if not self.recency:
            return
        self._rows_seen += len(rows)
        w = math.exp(self._rows_seen / self._n)
        self._w += w * cnt
        self._z += w * sm

    def _close(self, x: np.ndarray, labels: np.ndarray, c: np.ndarray, rec: dict) -> np.ndarray:
        if not self.recency:
            return super()._close(x, labels, c, rec)
        out = c.copy()
        live = self._w > 0
        out[live] = (self._z[live] / self._w[live, None]).astype(np.float32)
        return out
