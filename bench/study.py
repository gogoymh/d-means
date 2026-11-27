"""Where D-means' gain comes from: backbones, how many points to ask about, and which ones.

    uv run python -m bench.study --verify          # offline replay of results/study-clef.json
    uv run python -m bench.study --live --write    # ask Clef for missing answers, save results

Three experiments on the six datasets, three seeds, Clef as the decision model:

- backbone: D-means started from each baseline's own partition (K-means, and the official
  E-means and O-means), so its gain is measured against the very partition it starts from.
- budget: the share of points asked about, 0% to 100%, on the K-means backbone.
- selection: 30% of points chosen by smallest margin (the default) or at random.

The study keeps only the answers the main table does not already have, in
results/clef-study-answers.jsonl.gz, and reads the main Clef cache for the rest.
"""

import argparse
import gzip
import json
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

from dmeans import DMeans, _log
from dmeans.deciders import CacheMiss, Decisions
from dmeans.geometry import nearest
from dmeans.metrics import scores

from . import data, embed, official
from .semantic import DATASETS, MODEL, RESTARTS, RESULTS, SEEDS, paths

log = _log.get("bench.study")
OUT = RESULTS / "study-clef.json"
ANSWERS = RESULTS / "clef-study-answers.jsonl.gz"
BUDGETS = [0.0, 0.15, 0.30, 0.45, 0.60, 1.0]
BACKBONES = ["k-means", "e-means", "o-means"]


def decider(live: bool) -> tuple[Decisions, set[str]]:
    """The study's Decisions, preloaded with the main Clef answers; returns it and the main keys."""
    dec = Decisions(MODEL, ANSWERS, offline=not live)
    main = set()
    with gzip.open(paths(MODEL)[0], "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            main.add(row["k"])
            dec.store.setdefault(row["k"], row["a"])
            if row.get("refused"):
                dec.refused.add(row["k"])
    return dec, main


def backbone_centroids(name: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if name == "k-means":
        return KMeans(k, n_init=RESTARTS, random_state=seed).fit(x).cluster_centers_.astype(np.float32)
    return {"e-means": official.emeans, "o-means": official.omeans}[name](x, k, nepochs=5, seed=seed)[0]


def run(names: list[str], seeds: list[int], dec: Decisions) -> dict:
    out = {"model": MODEL, "embedder": embed.MODEL, "seeds": seeds, "budgets": BUDGETS, "datasets": {}}
    for name in names:
        ds = data.load(name)
        x = embed.embed(name, ds.texts)
        y = ds.labels
        res: dict[str, list[dict]] = {}

        def add(key: str, lab: np.ndarray, dm: DMeans | None = None) -> None:
            row = scores(y, lab)
            if dm is not None:
                row["decisions_per_point"] = dm.report["decisions_per_point"]
            res.setdefault(key, []).append(row)

        for seed in seeds:
            t0 = time.perf_counter()
            for b in BACKBONES:
                c = backbone_centroids(b, x, ds.k, seed)
                add(f"backbone {b}", nearest(x, c)[0])
                # The K-means backbone is D-means' default start: the same partition, the same questions.
                dm = DMeans(ds.k, dec, about=ds.about, seed=seed, start="kmeans" if b == "k-means" else c)
                add(f"d-means on {b}", dm.fit(x, ds.texts), dm)
            for budget in BUDGETS:
                dm = DMeans(ds.k, dec, about=ds.about, seed=seed, budget=budget)
                add(f"budget {budget:.2f}", dm.fit(x, ds.texts), dm)
            dm = DMeans(ds.k, dec, about=ds.about, seed=seed, select="random")
            add("random 30%", dm.fit(x, ds.texts), dm)
            log.info("%s seed %d done in %.1fs", name, seed, time.perf_counter() - t0)
        out["datasets"][name] = {"n": len(ds.texts), "k": ds.k, "runs": res, "mean": {m: {key: float(np.mean([r[key] for r in rs])) for key in rs[0]} for m, rs in res.items()}}
    return out


def verify(got: dict, ref: dict) -> list[str]:
    errors = []
    for name, d in got["datasets"].items():
        r = ref["datasets"].get(name)
        if r is None:
            errors.append(f"{name}: not in the committed results")
            continue
        for m, runs in d["runs"].items():
            for i, (a, b) in enumerate(zip(r["runs"].get(m, []), runs)):
                for key in ("acc", "nmi", "ari"):
                    if abs(a[key] - b[key]) > 1e-6:
                        errors.append(f"{name}/{m}/seed{i}: {key} {b[key]:.6f} != committed {a[key]:.6f}")
    return errors


def table(res: dict) -> str:
    lines = []
    for name, d in res["datasets"].items():
        lines.append(f"\n{name} (n={d['n']}, k={d['k']})")
        for m, v in d["mean"].items():
            lines.append(f"  {m:24s} ACC {100 * v['acc']:5.1f}  NMI {100 * v['nmi']:5.1f}  ARI {100 * v['ari']:5.1f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--live", action="store_true", help="ask Clef for answers missing from both caches")
    ap.add_argument("--write", action="store_true", help="save results/study-clef.json and the study's answers")
    ap.add_argument("--verify", action="store_true", help="fail unless results match results/study-clef.json")
    a = ap.parse_args(argv)
    _log.setup()
    dec, main_keys = decider(a.live)
    try:
        result = run(a.datasets.split(","), [int(s) for s in a.seeds.split(",")], dec)
    except CacheMiss as e:
        log.error("Clef answer %s is in neither cache; run with --live to ask it", e)
        return 2
    finally:
        if a.live:
            dec.save(keep=set(dec.store) - main_keys)
    log.info("%d requests, %d API calls, $%.4f", dec.stats["requests"], dec.stats["api_calls"], dec.stats["cost_usd"])
    print(table(result))
    if a.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        dec.save(keep=dec.used - main_keys)
        log.info("wrote %s", OUT)
    if a.verify:
        errors = verify(result, json.loads(OUT.read_text()))
        for e in errors:
            log.error("verify: %s", e)
        if errors or dec.stats["api_calls"]:
            return 1
        log.info("verify: results match %s with 0 API calls", OUT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
