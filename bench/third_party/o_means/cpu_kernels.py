# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU replacements for o-means' Triton kernels (triton_kernels.py, triton_two_level.py).

Each function keeps the official name, arguments and result, and reproduces what its kernel
computes: FP16 rows and centroids, FP32 scores cn - 2<x, c>, ties to the lowest index, batch
sums in int64 units of 2**-24, the same EMA and split arithmetic, and the same packed keys for
choosing probe cells. Only the work layout changes: plain torch on whole tensors instead of
tiles and atomics. build_cells, record_splits and permute_rows were already plain torch and are
copied verbatim (with next_power_of_2 inlined).
"""

import math
from dataclasses import dataclass

import torch

_FIXED_SCALE = 2.0**24
_CAST_HI, _CAST_LO = 2.0**15, 2.0**-8
EMPTY_NORM = 1e37


def next_power_of_2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _round_up(v, m):
    return (v + m - 1) // m * m


# --- triton_kernels.py ------------------------------------------------------------------------
@dataclass(frozen=True)
class DenseLayout:
    d: int
    k: int
    dp: int
    kp: int
    parts: int
    block_b: int = 64
    block_k: int = 256
    block_d: int = 32
    num_warps: int = 4
    num_stages: int = 2


def dense_layout(d, k, parts=8, block_b=64, block_k=256, block_d=32):
    parts = max(1, min(parts, -(-k // block_k)))
    return DenseLayout(d=d, k=k, dp=_round_up(d, block_d), kp=_round_up(k, parts * block_k), parts=parts, block_b=block_b, block_k=block_k, block_d=block_d)


def scale_exponent(max_abs):
    m = max_abs
    if not math.isfinite(m) or m == 0 or _CAST_LO <= m <= _CAST_HI:
        return 0
    return 7 - math.frexp(m)[1]


def cast_pad_rows(x, dp):
    """(x16, e): x16 = fp16(x * 2**e), zero-padded to dp columns."""
    n, d = x.shape
    e = scale_exponent(float(x.abs().max())) if n else 0
    out = torch.zeros((n, dp), dtype=torch.float16, device=x.device)
    out[:, :d] = x * 2.0**e if e else x
    return out, e


class DenseWorkspace:
    def __init__(self, rows, parts, device):
        self.rows, self.parts = rows, parts


def _rows(x16, idx, rows):
    if idx is not None:
        return x16[idx.long()]
    return x16[: (rows if rows is not None else x16.shape[0])]


def _scores(xr, c16, cn):
    """cn - 2 <x, c> in FP32 from the FP16 operands, as tl.dot(out_dtype=float32)."""
    return cn[None, :] - 2.0 * (xr.float() @ c16.float().t())


def assign_dense(x16, c16, cn, layout, ws, idx=None, rows=None):
    """Exact nearest-centroid labels, ties to the lowest centroid index."""
    xr = _rows(x16, idx, rows)
    if xr.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int64, device=x16.device)
    return _scores(xr, c16, cn).argmin(1)  # torch.argmin returns the first minimum


def accumulate(x16, labels, sums, cnt, d, idx=None):
    """sums[labels[i]] += x16[i] (int64, units of 2**-24), cnt[labels[i]] += 1."""
    R = int(labels.shape[0])
    if R == 0:
        return
    xr = _rows(x16, idx, R)[:, :d]
    v = (xr.float() * _FIXED_SCALE).to(torch.int64)  # exact: FP16 values are multiples of 2**-24
    lab = labels.long()
    sums[:, :d].index_add_(0, lab, v)
    cnt.index_add_(0, lab, torch.ones(R, dtype=cnt.dtype, device=cnt.device))


def ema_update(C, c16, cn, usage, sums, cnt, k, d, alpha, close_sums=None, close_cnt=None, omega=1.0):
    """One e-means update of every centroid that received rows (see the official docstring)."""
    close = close_sums is not None
    weighted = close and close_sums.dtype != sums.dtype
    replace = alpha == 1.0
    alpha32 = torch.tensor(alpha, dtype=torch.float32)
    decay32 = torch.tensor(1.0 - alpha, dtype=torch.float32)
    c_k = cnt[:k]
    touched = c_k > 0
    count = c_k.float()
    old = usage[:k].clone()
    retained = torch.zeros_like(old) if replace else old * decay32
    step = alpha32 * count
    mass = retained + step
    w = step / torch.where(touched, mass, torch.ones_like(mass))
    div = torch.where(touched, count, torch.ones_like(count))
    t = touched.nonzero().squeeze(1)
    if t.numel():
        s = sums[t, :d]
        sf = s.float() * (1.0 / _FIXED_SCALE)
        mean = sf / div[t, None]
        c = C[t, :d]
        wt = w[t]
        newc = torch.where((wt == 1.0)[:, None], mean, c + wt[:, None] * (mean - c))
        C[t, :d] = newc
        h = newc.half()
        c16[t, :d] = h
        cn[t] = h.float().square().sum(1)
        if close:
            if weighted:
                close_sums[t, :d] += torch.tensor(omega, dtype=torch.float32) * sf
                close_cnt[t] += torch.tensor(omega, dtype=torch.float32) * count[t]
            else:
                close_sums[t, :d] += s
                close_cnt[t] += c_k[t].to(close_cnt.dtype)
        sums[t, :d] = 0
        cnt[t] = 0
    usage[:k] = torch.where(touched, mass, old if replace else retained)


def refresh_table(C, c16, cn, k, d):
    c16[:k, :d] = C[:k, :d]
    cn[:k] = c16[:k, :d].float().square().sum(1)


def _nudge_fp16(h):
    """Move FP16 values one step, as _split_kernel does for a child equal to its donor."""
    b = h.view(torch.int16).to(torch.int32) & 0xFFFF
    mag = b & 0x7FFF
    neg = (b & 0x8000) != 0
    up = torch.where(mag == 0, torch.ones_like(b), torch.where(neg, b - 1, b + 1))
    inward = torch.where(mag == 0, torch.full_like(b, 0x8001), torch.where(neg, b + 1, b - 1))
    moved = torch.where((up & 0x7FFF) < 0x7C00, up, inward)
    moved = torch.where(mag > 0x7C00, b, moved)  # NaN stays NaN
    moved = moved & 0xFFFF
    moved = torch.where(moved >= 0x8000, moved - 0x10000, moved)
    return moved.to(torch.int16).view(torch.float16)


def apply_splits(C, c16, cn, usage, src, tgt, m, d, eps, max_pairs, wsum=None, wcnt=None):
    """Split donor src[i] into itself and child tgt[i], for i < m."""
    m = int(m)
    if m <= 0:
        return
    s, t = src[:m].long(), tgt[:m].long()
    v = C[s, :d].clone()
    col = torch.arange(d, device=C.device)
    off = torch.tensor(eps, dtype=torch.float32) * torch.where(col % 2 == 0, 1.0, -1.0).float()
    lo = v * (1.0 - off)
    hi = v * (1.0 + off)
    hlo, hhi = lo.half(), hi.half()
    ndiff = (hlo != hhi).sum(1)
    C[s, :d] = lo
    C[t, :d] = hi
    c16[s, :d] = hlo
    c16[t, :d] = hhi
    norm_lo = hlo.float().square().sum(1)
    norm_hi = hhi.float().square().sum(1)
    if wsum is not None:
        hw = wsum[s, :d] * 0.5
        wsum[s, :d] = hw
        wsum[t, :d] = hw
    same = ndiff == 0
    if same.any():
        ts = t[same]
        h0 = c16[ts, 0]
        nb = _nudge_fp16(h0)
        f_old, f_new = h0.float(), nb.float()
        norm_hi[same] = norm_hi[same] + f_new * f_new - f_old * f_old
        c16[ts, 0] = nb
        C[ts, 0] = f_new
    cn[s] = norm_lo
    cn[t] = norm_hi
    half = usage[s] * 0.5
    usage[s] = half
    usage[t] = half
    if wcnt is not None:
        hc = wcnt[s] * 0.5
        wcnt[s] = hc
        wcnt[t] = hc


# --- triton_two_level.py ----------------------------------------------------------------------
@dataclass
class Cells:
    G: int
    GP: int
    g32: torch.Tensor
    g16: torch.Tensor
    gn: torch.Tensor
    off: torch.Tensor
    size: torch.Tensor
    child: torch.Tensor


class TwoLevelWorkspace:
    def __init__(self, rows, G, max_p, device):
        self.rows, self.G, self.max_p = rows, G, max_p


def _pack_keys(score, GP):
    """Monotone int32 keys of the scores, low bits replaced by the cell id (as _pack_keys)."""
    bits = score.contiguous().view(torch.int32)
    key = torch.where(bits < 0, bits ^ 2147483647, bits)
    gid = torch.arange(score.shape[1], dtype=torch.int32, device=score.device)
    return (key & (-GP)) | gid[None, :]


def two_level_assign(x16, c16, cn, cells, ws, p, idx=None, rows=None):
    """Labels as positions in the cell-sorted table: the best centroid of the p nearest cells,
    then the split children of that winner (as _coarse, _fine and _reduce_child kernels)."""
    xr = _rows(x16, idx, rows)
    R = xr.shape[0]
    if R == 0:
        return torch.zeros(0, dtype=torch.int64, device=x16.device)
    G = cells.G
    key = _pack_keys(_scores(xr, cells.g16, cells.gn), cells.GP)
    probes = key.argsort(1)[:, :p]  # keys are distinct; rank 0 is the nearest cell
    score = _scores(xr, c16, cn)
    k_used = int(cells.off[G])
    cell_of = torch.repeat_interleave(torch.arange(G, device=x16.device), cells.size.long())
    sc = score[:, :k_used]
    best = torch.full((R,), float("inf"), dtype=torch.float32, device=x16.device)
    lab = torch.zeros(R, dtype=torch.int64, device=x16.device)
    for r in range(p):  # strict: the lower probe rank wins ties
        masked = torch.where(cell_of[None, :] == probes[:, r : r + 1], sc, torch.tensor(float("inf")))
        bl = masked.argmin(1)  # first minimum: ties to the lowest index; an empty cell gives 0, as the kernel
        bs = masked.gather(1, bl[:, None]).squeeze(1)
        better = bs < best
        best = torch.where(better, bs, best)
        lab = torch.where(better, bl, lab)
    child = cells.child.long()
    c1 = child[lab]
    c1 = torch.where(c1 < 0, lab, c1)
    c2 = child[c1]
    c2 = torch.where(c2 < 0, c1, c2)
    n1, n2 = c1 != lab, c2 != c1
    xf = xr.float()
    s1 = torch.where(n1, cn[c1] - 2.0 * (xf * c16[c1].float()).sum(1), torch.tensor(float("inf")))
    s2 = torch.where(n2, cn[c2] - 2.0 * (xf * c16[c2].float()).sum(1), torch.tensor(float("inf")))
    t1 = n1 & (s1 < best)
    best = torch.where(t1, s1, best)
    lab = torch.where(t1, c1, lab)
    t2 = n2 & (s2 < best)
    lab = torch.where(t2, c2, lab)
    return lab


def build_cells(C, k, G, iters, prev=None, generator=None):
    """Group the centroids C[:k] into G cells by Lloyd iterations on them (official, verbatim)."""
    dev = C.device
    dp = C.shape[1]
    Ck = C[:k]
    if prev is None:
        pick = torch.randperm(k, generator=generator, device=dev)[:G]
        g = Ck[pick].clone()
    else:
        g = prev.g32.clone()
    ones = torch.ones(k, dtype=torch.float32, device=dev)
    for _ in range(iters):
        a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
        sums = torch.zeros((G, dp), dtype=torch.float32, device=dev).index_add_(0, a, Ck)
        cnt = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, a, ones)
        g = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1)[:, None], g)
    a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
    order = torch.argsort(a, stable=True)
    size = torch.bincount(a, minlength=G)
    off = torch.zeros(G + 1, dtype=torch.int32, device=dev)
    off[1:] = torch.cumsum(size, 0)
    GP = max(16, next_power_of_2(G))
    g16 = torch.zeros((GP, dp), dtype=torch.float16, device=dev)
    g16[:G] = g
    gn = torch.full((GP,), float("inf"), dtype=torch.float32, device=dev)
    gn[:G] = torch.where(size > 0, g16[:G].float().square().sum(1), EMPTY_NORM)
    child = torch.full((C.shape[0] + 1,), -1, dtype=torch.int32, device=dev)
    return order, Cells(G=G, GP=GP, g32=g, g16=g16, gn=gn, off=off, size=size, child=child)


def record_splits(child, src, tgt, m, max_pairs):
    """child[t] = child[s]; child[s] = t for the splits of one revival check (official, verbatim)."""
    spare = child.shape[0] - 1
    s, t = src[:max_pairs], tgt[:max_pairs]
    live = torch.arange(s.shape[0], device=s.device) < m
    s = torch.where(live, s, spare)
    t = torch.where(live, t, spare)
    inherited = child[s]
    child[t] = inherited
    child[s] = t.to(child.dtype)
    child[spare] = -1


def permute_rows(order, k, tensors):
    """t[:k] = t[:k][order] in place for each tensor (official, verbatim)."""
    for t in tensors:
        t[:k] = t[:k].index_select(0, order)
