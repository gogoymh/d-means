"""Engine benchmark: held-out MSE of Lloyd's k-means, e-means and o-means (no decision model).

It checks that the e-means and o-means reimplementations behave as their authors report:
e-means passes Lloyd's 25-iteration error within a few epochs, and o-means reaches the same
error while computing a fraction of the point-to-centroid distances. d-means runs on this
engine, so it inherits the speed.

    uv run python -m bench.mse --write
"""

import argparse
import json
import sys
import time

import numpy as np

from dmeans import EMeans, OMeans, _log, lloyd
from dmeans.geometry import mse

from . import data, embed
from .semantic import RESULTS

log = _log.get("bench.mse")
OUT = RESULTS / "mse.json"
EPOCHS = [1, 2, 3, 5]


def gaussian_mixture(n: int, d: int, modes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(modes, d)).astype(np.float32)
    return centers[rng.integers(0, modes, n)] + 0.6 * rng.normal(size=(n, d)).astype(np.float32)


def problems() -> dict:
    rng = np.random.default_rng(0)
    syn = gaussian_mixture(105_000, 32, 16_384)
    real = np.concatenate([embed.embed(n, data.load(n).texts) for n in data.SOURCES])
    real = real[rng.permutation(len(real))]
    return {
        "synthetic (n=100k, d=32, k=8192)": (syn[:100_000], syn[100_000:], 8192),
        "text embeddings (n=20k, d=768, k=512)": (real[:20_000], real[20_000:], 512),
    }


def bench(xtr: np.ndarray, xte: np.ndarray, k: int, seed: int) -> dict:
    rows = {}
    curve = []
    t0 = time.perf_counter()
    lloyd(xtr, k, iters=25, seed=seed, on_epoch=lambda it, c: curve.append((it, time.perf_counter() - t0, mse(xte, c))))
    ref = curve[-1][2]
    rows["Lloyd"] = [{"epochs": it, "seconds": s, "test_mse": m} for it, s, m in curve]
    for cls in (EMeans, OMeans):
        out = []
        for t in EPOCHS:
            m = cls(k, nepochs=t, seed=seed)
            t1 = time.perf_counter()
            c, _ = m.fit(xtr)
            sec = time.perf_counter() - t1
            out.append({"epochs": t, "seconds": sec, "test_mse": mse(xte, c), "distance_evals": m.distance_evals, "recipe": {k_: v for k_, v in m.recipe.items()}})
            log.info("%s T=%d: mse %.5f (Lloyd@25 %.5f) in %.1fs", cls.name, t, out[-1]["test_mse"], ref, sec)
        rows[cls.name] = out
    for name in ("e-means", "o-means"):
        for r in rows[name]:
            r["mse_vs_lloyd25"] = r["test_mse"] / ref - 1.0
    for e, o in zip(rows["e-means"], rows["o-means"]):
        o["distance_evals_vs_emeans"] = o["distance_evals"] / e["distance_evals"]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    _log.setup()
    result = {name: bench(xtr, xte, k, seed=1) for name, (xtr, xte, k) in problems().items()}
    for name, rows in result.items():
        print(f"\n{name}")
        print(f"  Lloyd@25: test MSE {rows['Lloyd'][-1]['test_mse']:.5f} in {rows['Lloyd'][-1]['seconds']:.1f}s")
        for m in ("e-means", "o-means"):
            for r in rows[m]:
                extra = f", distances x{r['distance_evals_vs_emeans']:.2f} of e-means" if "distance_evals_vs_emeans" in r else ""
                print(f"  {m}@{r['epochs']}: {r['mse_vs_lloyd25']:+.2%} vs Lloyd@25 in {r['seconds']:.1f}s{extra}")
    if a.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
