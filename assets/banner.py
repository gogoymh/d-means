"""Render a letter D of small spheres, clustered at its corners, next to the D-means wordmark.

    uv run python assets/banner.py          # writes d-means-banner.png and d-means-banner-dark.png
    uv run python assets/banner.py OUTDIR   # writes them into OUTDIR instead

Pure numpy: an analytic ray tracer (soft shadows, ambient occlusion, satin reflections)
and a small PNG codec, so the banner needs no image library and no licensed art.
"""

from __future__ import annotations

import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from dmeans import _log

HERE = Path(__file__).resolve().parent
LOG = _log.get("banner")

W, H, SS = 1536, 512, 3  # output size and supersampling per axis
WORDMARK_AT = (773, 206)  # top-left of assets/wordmark.png on the banner

# Studio camera: eye, target, focal length (px) and where the target lands on the banner.
EYE, TARGET, FOCAL, CENTER = (0.0, 7.0, 10.0), (0.0, 0.25, 0.0), 1120.0, (373.0, 263.0)

KEY = np.array([-0.55, 1.0, 0.35]) / np.linalg.norm([-0.55, 1.0, 0.35])  # softbox, upper left
KEY_RADIUS = 0.42  # angular radius (rad) of the softbox: wide, soft shadows
STRIP = np.array([1.0, 0.35, 0.15]) / np.linalg.norm([1.0, 0.35, 0.15])  # thin strip light, right
REFLECTED = 8  # neighbours each sphere reflects
RIM = np.array([0.2, 0.45, -1.0]) / np.linalg.norm([0.2, 0.45, -1.0])  # back light (dark theme)

# The letter is laid out on screen, then each sphere is placed on the floor under its spot, so the D
# reads upright from the camera. Three clusters gather at its corners (top-left, bottom-left and the
# tip of the bowl); every sphere takes the colour of the nearest centroid, as k-means would.
D_BOX = (262, 118, 490, 404)  # left, top, right, bottom of the letter on the banner (px)
D_CENTROIDS = (("amber", (262, 118)), ("violet", (262, 404)), ("teal", (490, 261)))
D_SEED = 11


def d_outline(step: float = 1.0) -> np.ndarray:
    """Points every `step` px along the outline of a D: stem, top edge, bowl, bottom edge."""
    left, top, right, bottom = D_BOX
    r = (bottom - top) / 2
    cx, cy = right - r, (top + bottom) / 2
    stem = [(left, y) for y in np.arange(bottom, top, -step)]
    upper = [(x, top) for x in np.arange(left, cx, step)]
    a = np.arange(-np.pi / 2, np.pi / 2, step / r)
    bowl = list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))
    lower = [(x, bottom) for x in np.arange(cx, left, -step)]
    return np.array(stem + upper + bowl + lower, float)


def d_layout():
    """Screen-space spheres (material, px, py, radius px): centroids first, then members."""
    rng = np.random.default_rng(D_SEED)
    centers = np.array([c for _, c in D_CENTROIDS], float)
    placed = [(name, x, y, 15.0) for name, (x, y) in D_CENTROIDS]

    def free(x, y, r):
        return all(np.hypot(x - px, y - py) > r + pr + 3.0 for _, px, py, pr in placed)

    def add(x, y, r):
        if free(x, y, r):
            name = D_CENTROIDS[int(np.argmin(np.hypot(*(centers - (x, y)).T)))][0]
            placed.append((name, x, y, r))

    # along the outline: tight near a centroid, sparse in between
    path = d_outline()
    s = 0.0
    while s < len(path):
        x, y = path[int(s)]
        near = np.hypot(*(centers - (x, y)).T).min()
        add(x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), rng.uniform(6.0, 8.0))
        s += 23 - 7 * np.exp(-((near / 60) ** 2))
    # a crowd of extra members around each centroid
    for _, (cx, cy) in D_CENTROIDS:
        for _ in range(60):
            a, d = rng.uniform(0, 2 * np.pi), abs(rng.normal(0, 13)) + 24
            add(cx + d * np.cos(a), cy + d * np.sin(a), rng.uniform(5.0, 7.0))
    return placed


def floor_spheres(layout):
    """Back-project screen spheres onto the floor: (material, x, z, radius) in world units."""
    eye = np.array(EYE)
    fwd = norm(np.array(TARGET) - eye)
    right = norm(np.cross(fwd, [0.0, 1.0, 0.0]))
    up = np.cross(right, fwd)
    out = []
    for name, px, py, rp in layout:
        d = norm((px - CENTER[0]) / FOCAL * right - (py - CENTER[1]) / FOCAL * up + fwd)
        r = 0.0
        for _ in range(4):  # the sphere rests on the floor, so its centre sits at height r
            p = eye + d * (r - eye[1]) / d[1]
            r = rp * ((p - eye) @ fwd) / FOCAL
        out.append((name, float(p[0]), float(p[2]), float(r)))
    return tuple(out)


# Backdrop colours (sRGB), wordmark ink, diffuse light (key, sky, ground), the environment the
# metal reflects (zenith, horizon, floor, softbox, strip, rim) and how much the floor darkens.
THEMES = {
    "light": dict(
        center=(244, 245, 247), edge=(212, 214, 218), ink=(27, 29, 33),
        key=0.75, sky=0.42, ground=0.22,
        zenith=0.82, horizon=0.30, floor=0.60, softbox=3.0, strip=1.0, rim=0.0,
        shadow=0.55, ao=0.85,
    ),
    "dark": dict(
        center=(36, 38, 43), edge=(11, 12, 14), ink=(241, 242, 244),
        key=0.75, sky=0.10, ground=0.02,
        zenith=0.24, horizon=0.02, floor=0.035, softbox=2.6, strip=0.8, rim=1.2,
        shadow=0.75, ao=0.9,
    ),
}

# Linear albedo / F0 and roughness per material.
MATERIALS = {
    "amber": dict(albedo=(0.83, 0.36, 0.075), f0=0.04, rough=0.42, wrap=0.32),
    "teal": dict(albedo=(0.035, 0.46, 0.38), f0=0.04, rough=0.42, wrap=0.32),
    "violet": dict(albedo=(0.31, 0.16, 0.65), f0=0.04, rough=0.42, wrap=0.32),
}


# ---------------------------------------------------------------- colour helpers

def to_linear(srgb) -> np.ndarray:
    c = np.asarray(srgb, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def shoulder(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Roll highlights off above `knee` instead of clipping them flat."""
    over = np.maximum(x - knee, 0.0)
    return np.minimum(x, knee) + (1 - knee) * over / (over + (1 - knee))


def dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- scene

SPHERES = floor_spheres(d_layout())


def spheres():
    c = np.array([(x, r, z) for _, x, z, r in SPHERES], float)
    r = np.array([s[3] for s in SPHERES], float)
    return c, r


def background(u, v, t):
    """Seamless studio backdrop in screen space (linear), brightest behind the wordmark."""
    d = np.sqrt(((u - 0.60) / 0.80) ** 2 + ((v - 0.58) / 0.95) ** 2)
    k = smoothstep(0.0, 1.0, d)[..., None]
    return to_linear(t["center"]) * (1 - k) + to_linear(t["edge"]) * k


def lobe(d, axis, size, rough):
    """Soft area light of angular size, widened by roughness with its energy kept."""
    s2 = size**2 + (0.6 * rough) ** 2
    ang = np.arccos(np.clip(dot(d, axis), -1, 1))[..., None]
    return np.exp(-(ang**2) / (2 * s2)) * size**2 / s2


def env(d, rough, t):
    """Studio environment seen along d: floor, a darker horizon band, a bright zenith, lights."""
    y = d[..., 1:2]
    sky = t["horizon"] + (t["zenith"] - t["horizon"]) * np.clip(y, 0, 1) ** 0.8
    k = smoothstep(-0.08 - 0.5 * rough, 0.08 + 0.5 * rough, y)
    rad = (t["floor"] * (1 - k) + sky * k) * np.ones(3)
    rad = rad + t["softbox"] * lobe(d, KEY, KEY_RADIUS * 0.55, rough) + t["strip"] * lobe(d, STRIP, 0.12, rough)
    if t["rim"]:
        rad = rad + t["rim"] * lobe(d, RIM, 0.35, rough)
    return rad


def key_visibility(p, centers, radii, skip=None):
    """Fraction of the softbox seen from p: analytic disc overlap per sphere (no noise)."""
    vis = np.ones(p.shape[:-1])
    a = KEY_RADIUS
    for i, (c, r) in enumerate(zip(centers, radii)):
        v = c - p
        dist = np.linalg.norm(v, axis=-1)
        b = np.arcsin(np.clip(r / np.maximum(dist, r), 0, 1))
        th = np.arccos(np.clip(dot(v, KEY) / dist, -1, 1))
        # area of intersection of two discs (radii a, b, centre distance th) / area of a
        d = np.maximum(th, 1e-6)
        ca = np.clip((d * d + a * a - b * b) / (2 * d * a), -1, 1)
        cb = np.clip((d * d + b * b - a * a) / (2 * d * b), -1, 1)
        lens = a * a * np.arccos(ca) + b * b * np.arccos(cb) - 0.5 * np.sqrt(
            np.clip((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b), 0, None))
        inter = np.where(th >= a + b, 0.0, np.where(th <= np.abs(a - b), np.pi * np.minimum(a, b) ** 2, lens))
        frac = np.clip(inter / (np.pi * a * a), 0, 1)
        if skip is not None:
            frac = np.where(skip == i, 0.0, frac)
        vis = vis * (1 - frac)
    return vis


def occlusion(p, n, centers, radii, skip=None):
    """Cosine-weighted ambient occlusion by the spheres (Quilez's analytic sphere AO)."""
    ao = np.ones(p.shape[:-1])
    for i, (c, r) in enumerate(zip(centers, radii)):
        v = c - p
        dist = np.linalg.norm(v, axis=-1)
        occ = np.clip(dot(n, v) / dist, 0, 1) * (r / np.maximum(dist, r)) ** 2
        if skip is not None:
            occ = np.where(skip == i, 0.0, occ)
        ao = ao * (1 - occ)
    return ao


def hit_spheres(o, d, centers, radii):
    """Nearest sphere hit per ray: (t, index); index is -1 on a miss."""
    best = np.full(d.shape[:-1], np.inf)
    idx = np.full(d.shape[:-1], -1)
    for i, (c, r) in enumerate(zip(centers, radii)):
        oc = o - c
        b = dot(oc, d)
        disc = b * b - dot(oc, oc) + r * r
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        ok = (disc > 0) & (t > 1e-4) & (t < best)
        best = np.where(ok, t, best)
        idx = np.where(ok, i, idx)
    return best, idx


def reflect_spheres(o, d, me, rough, spec, centers, radii, t):
    """Blend the other spheres into a satin reflection, with edges blurred by roughness."""
    near = sorted((j for j in range(len(radii)) if j != me), key=lambda j: np.linalg.norm(centers[j] - centers[me]))
    for j in near[:REFLECTED][::-1]:  # only close neighbours show in satin; far ones first
        c, r = centers[j], radii[j]
        oc = o - c
        b = dot(oc, d)
        miss2 = np.maximum(dot(oc, oc) - b * b, 0.0)
        w = np.maximum(0.8 * rough * -b, 1e-4)
        k = np.where(-b > 0, smoothstep(r + w, r - w, np.sqrt(miss2)), 0.0)
        on = k > 1e-3
        if not on.any():
            continue
        o_, d_, b_ = o[on], d[on], b[on]
        disc = r * r - miss2[on]
        near = o_ + d_ * (-b_)[..., None]  # closest approach; projected onto the sphere on a near miss
        q = np.where((disc > 0)[..., None], o_ + d_ * (-b_ - np.sqrt(np.maximum(disc, 0)))[..., None],
                     c + r * (near - c) / np.maximum(np.linalg.norm(near - c, axis=-1, keepdims=True), 1e-9))
        bounce = shade_sphere(q, norm(q - c), -d_, np.full(q.shape[0], j), centers, radii, t, depth=1)
        spec[on] = spec[on] * (1 - k[on][..., None]) + bounce * k[on][..., None]
    return spec


def shade_sphere(p, n, v, idx, centers, radii, t, depth=0):
    """Radiance leaving sphere points p (normal n) toward the viewer direction v."""
    out = np.zeros(p.shape)
    ndv = np.clip(dot(n, v), 1e-4, 1)[..., None]
    refl = norm(2 * ndv * n - v)
    vis = key_visibility(p + n * 1e-3, centers, radii, skip=idx)[..., None]
    ao = occlusion(p + n * 1e-3, n, centers, radii, skip=idx)[..., None]
    for i, (name, *_rest) in enumerate(SPHERES):
        m = MATERIALS[name]
        sel = idx == i
        if not sel.any():
            continue
        nn, rr, vv, aa = n[sel], refl[sel], vis[sel], ao[sel]
        f0 = np.asarray(m["f0"], float) * np.ones(3)
        c = ndv[sel]
        fres = f0 + (np.maximum(1 - m["rough"], f0) - f0) * (1 - c) ** 5
        # diffuse: wrapped Lambert from the softbox plus hemispherical sky / floor light
        ndl = dot(nn, KEY)[..., None]
        s = np.sin(KEY_RADIUS) + m["wrap"]
        ndl = np.where(ndl > s, ndl, (np.clip(ndl, -s, s) + s) ** 2 / (4 * s))
        hemi = t["ground"] + (t["sky"] - t["ground"]) * (0.5 + 0.5 * nn[..., 1:2])
        diffuse = np.asarray(m["albedo"]) * (t["key"] * ndl * vv + hemi * aa)
        # specular: blurred environment, or the other spheres it reflects
        spec = env(rr, m["rough"], t)
        if depth == 0:
            spec = reflect_spheres(p[sel] + nn * 1e-3, rr, i, m["rough"], spec, centers, radii, t)
            spec = spec * np.clip(aa * 1.15, 0, 1)
        out[sel] = (1 - fres) * diffuse + fres * spec
    return out


class Camera:
    def __init__(self):
        self.eye = np.array(EYE)
        self.fwd = norm(np.array(TARGET) - self.eye)
        self.right = norm(np.cross(self.fwd, [0.0, 1.0, 0.0]))
        self.up = np.cross(self.right, self.fwd)

    def rays(self, px, py):
        d = (px[..., None] - CENTER[0]) / FOCAL * self.right - (py[..., None] - CENTER[1]) / FOCAL * self.up
        return np.broadcast_to(self.eye, d.shape), norm(d + self.fwd)

    def sphere_bounds(self, centers, radii):
        """Pixel bounds containing all projected spheres, from their camera-space boxes."""
        view = centers - self.eye
        offsets = radii[:, None] * np.array([-1, 1])
        depth = (view @ self.fwd)[:, None] + offsets
        x = CENTER[0] + FOCAL * ((view @ self.right)[:, None] + offsets)[:, :, None] / depth[:, None, :]
        y = CENTER[1] - FOCAL * ((view @ self.up)[:, None] + offsets)[:, :, None] / depth[:, None, :]
        return (
            max(0, int(np.floor(x.min()))), min(W, int(np.ceil(x.max()))),
            max(0, int(np.floor(y.min()))), min(H, int(np.ceil(y.max()))),
        )


def shade(px, py, cam, t, centers, radii):
    """Linear radiance of the rays through banner positions (px, py)."""
    o, d = cam.rays(px, py)
    col = background(px / W, py / H, t)
    # floor: the backdrop darkened by contact occlusion and the soft key shadow
    down = d[..., 1] < 0
    tf = np.where(down, -o[..., 1] / np.where(down, d[..., 1], -1.0), 0.0)
    pf = o + d * tf[..., None]
    nf = np.broadcast_to(np.array([0.0, 1.0, 0.0]), d.shape)
    ao = occlusion(pf, nf, centers, radii)
    vis = key_visibility(pf, centers, radii)
    dim = (1 - t["ao"] * (1 - ao)) * (1 - t["shadow"] * (1 - vis))
    col = col * np.where(down, dim, 1.0)[..., None]
    ts, idx = hit_spheres(o, d, centers, radii)
    hit = idx >= 0
    if hit.any():
        p = o[hit] + d[hit] * ts[hit][..., None]
        n = norm(p - centers[idx[hit]])
        col[hit] = shoulder(shade_sphere(p, n, -d[hit], idx[hit], centers, radii, t))
    return col


def render(theme: str = "light") -> np.ndarray:
    """Banner as an (H, W, 3) uint8 sRGB array. Silhouette pixels get SS x SS samples."""
    t = THEMES[theme]
    started = time.perf_counter()
    centers, radii = spheres()
    cam = Camera()
    col = np.zeros((H, W, 3))
    edge = np.zeros((H, W), bool)
    offs = (np.arange(SS) + 0.5) / SS
    left, right, top, bottom = cam.sphere_bounds(centers, radii)
    sx = (np.arange(left, right)[:, None] + offs).reshape(-1)
    rows = 64
    for y0 in range(0, H, rows):
        py, px = np.mgrid[y0: y0 + rows, 0:W] + 0.5
        col[y0: y0 + rows] = shade(px, py, cam, t, centers, radii)
        start, end = max(y0, top), min(y0 + rows, bottom)
        if start < end:
            sy = (np.arange(start, end)[:, None] + offs).reshape(-1)
            o, d = cam.rays(*np.meshgrid(sx, sy))
            idx = hit_spheres(o, d, centers, radii)[1].reshape(end - start, SS, right - left, SS)
            edge[start:end, left:right] = idx.min(axis=(1, 3)) != idx.max(axis=(1, 3))
    ey, ex = np.nonzero(edge)
    sub_x = (ex[:, None, None] + offs[None, None, :]) * np.ones((1, SS, 1))
    sub_y = (ey[:, None, None] + offs[None, :, None]) * np.ones((1, 1, SS))
    col[ey, ex] = shade(sub_x.reshape(-1), sub_y.reshape(-1), cam, t, centers, radii).reshape(-1, SS * SS, 3).mean(1)
    srgb = to_srgb(col)
    # the wordmark is composited in display space, the way it was first drawn
    alpha = png_read(HERE / "wordmark.png")[..., 0, None] / 255.0
    x0, y0 = WORDMARK_AT
    box = srgb[y0: y0 + alpha.shape[0], x0: x0 + alpha.shape[1]]
    box[:] = box * (1 - alpha) + np.asarray(t["ink"], float) * alpha
    LOG.info("rendered %s banner %dx%d, %d silhouette px at %dx%d, in %.1fs",
             theme, W, H, ey.size, SS, SS, time.perf_counter() - started)
    return np.clip(np.round(srgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- PNG codec (8-bit, filters 0-2)

def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_write(path: Path, img: np.ndarray) -> int:
    img = np.ascontiguousarray(img, np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    h, w, ch = img.shape
    rows = img.reshape(h, w * ch).astype(np.int16)
    left = np.hstack([np.zeros((h, ch), np.int16), rows[:, :-ch]])
    above = np.vstack([np.zeros((1, w * ch), np.int16), rows[:-1]])
    cands = np.stack([rows, (rows - left) & 255, (rows - above) & 255])  # None, Sub, Up
    cost = np.abs(np.where(cands > 127, cands - 256, cands)).sum(axis=2)
    pick = cost.argmin(axis=0)
    body = np.hstack([pick[:, None], cands[pick, np.arange(h)]]).astype(np.uint8).tobytes()
    head = struct.pack(">IIBBBBB", w, h, 8, {1: 0, 3: 2, 4: 6}[ch], 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", head) + _chunk(b"IDAT", zlib.compress(body, 9)) + _chunk(b"IEND", b"")
    Path(path).write_bytes(data)
    return len(data)


def png_read(path: Path) -> np.ndarray:
    data, pos, idat = Path(path).read_bytes(), 8, b""
    while pos < len(data):
        n, kind = struct.unpack(">I", data[pos: pos + 4])[0], data[pos + 4: pos + 8]
        if kind == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", data[pos + 8: pos + 21])
        elif kind == b"IDAT":
            idat += data[pos + 8: pos + 8 + n]
        pos += 12 + n
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    if depth != 8 or interlace:
        raise ValueError(f"unsupported PNG: depth={depth} interlace={interlace}")
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * ch)
    out = np.zeros((h, w * ch), np.uint8)
    prev = np.zeros(w * ch, np.uint8)
    for y in range(h):
        f, row = raw[y, 0], raw[y, 1:]
        if f == 0:
            cur = row
        elif f == 1:
            cur = (row.reshape(w, ch).astype(np.int64).cumsum(axis=0) & 255).astype(np.uint8).reshape(-1)
        elif f == 2:
            cur = row + prev
        else:
            cur = _unfilter_slow(f, row, prev, ch)
        out[y] = prev = cur
    return out.reshape(h, w, ch)


def _unfilter_slow(f, row, prev, ch):
    cur = np.zeros(row.size, np.int64)
    for k in range(row.size):
        a = int(cur[k - ch]) if k >= ch else 0
        b = int(prev[k])
        c = int(prev[k - ch]) if k >= ch else 0
        if f == 3:
            pred = (a + b) // 2
        else:
            pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
            pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        cur[k] = (int(row[k]) + pred) & 255
    return cur.astype(np.uint8)


def main(argv: list[str]) -> None:
    _log.setup()
    outdir = Path(argv[0]) if argv else HERE
    outdir.mkdir(parents=True, exist_ok=True)
    for theme, name in (("light", "d-means-banner.png"), ("dark", "d-means-banner-dark.png")):
        size = png_write(outdir / name, render(theme))
        LOG.info("wrote %s (%d KB)", outdir / name, size // 1024)


if __name__ == "__main__":
    main(sys.argv[1:])
