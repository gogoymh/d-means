"""Semantic clustering benchmark: k-means vs e-means vs o-means vs d-means on labeled texts.

    uv run python -m bench.semantic            # replay Jev's cached answers, print the table
    uv run python -m bench.semantic --verify   # also check it matches results/semantic.json
    uv run python -m bench.semantic --live --write   # ask Jev for missing answers, save results

Replay is offline and exact: d-means sends the same requests, and the committed cache returns
the answers Jev gave when the benchmark was run live. A request missing from the cache fails
the replay instead of being guessed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from dmeans import DMeans, EMeans, Jev, OMeans, _log
from dmeans.geometry import nearest
from dmeans.jev import CacheMiss
from dmeans.metrics import scores

from . import data, embed

log = _log.get("bench.semantic")
RESULTS = Path(__file__).resolve().parent / "results"
ANSWERS = RESULTS / "jev-answers.jsonl.gz"
OUT = RESULTS / "semantic.json"
DATASETS = ["trec", "stackexchange"]  # measured with live Jev answers
SEEDS = [0, 1, 2]
BASELINES = ["k-means", "e-means", "o-means"]
# Cost per Jev decision measured on the live runs (OpenRouter usage.cost, Jev 1.14):
# 447 requests cost $0.01175 on trec and 3,684 requests cost $0.09766 on stackexchange.
USD_PER_DECISION = round((0.01175 + 0.09766) / (447 + 3684), 8)

ABLATIONS = {
    "d-means": {},
    "merge only (no point decisions)": {"budget": 0.0},
    "point budget 5%": {"budget": 0.05},
    "point budget 10%": {"budget": 0.10},
    "decision only (no geometric prior)": {"fusion": "decision"},
    "10 merge neighbours": {"neighbors": 10},
    "spectral merge": {"linkage": "spectral"},
}


def geometric_merge(x: np.ndarray, k: int, seed: int, over: int = 3) -> np.ndarray:
    """d-means without decisions: same micro-clusters, merged by average-linkage on cosine."""
    k0 = int(min(over * k, max(k, len(x) // 4)))
    mc, _ = EMeans(k0, nepochs=5, seed=seed).fit(x)
    ml, _ = nearest(x, mc)
    group = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(mc)
    return group[ml]


def run_baseline(method: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "k-means":
        return KMeans(k, n_init=1, random_state=seed).fit(x).labels_
    model = {"e-means": EMeans, "o-means": OMeans}[method](k, nepochs=5, seed=seed)
    c, _ = model.fit(x)
    return nearest(x, c)[0]


def _mean(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def run(datasets: list[str], seeds: list[int], jev: Jev, ablations: bool = True) -> dict:
    out = {
        "jev_model": jev.model,
        "embedder": embed.MODEL,
        "seeds": seeds,
        "usd_per_decision": USD_PER_DECISION,
        "datasets": {},
    }
    for name in datasets:
        ds = data.load(name)
        x = embed.embed(name, ds.texts)
        y = ds.labels
        res: dict[str, list[dict]] = {}
        for seed in seeds:
            for m in BASELINES:
                t0 = time.perf_counter()
                lab = run_baseline(m, x, ds.k, seed)
                res.setdefault(m, []).append({**scores(y, lab), "seconds": time.perf_counter() - t0})
            if ablations:
                lab = geometric_merge(x, ds.k, seed)
                res.setdefault("geometric merge (no decisions)", []).append(scores(y, lab))
            for label, kw in (ABLATIONS if ablations else {"d-means": {}}).items():
                dm = DMeans(ds.k, jev, about=ds.about, seed=seed, **kw)
                t0 = time.perf_counter()
                lab = dm.fit(x, ds.texts)
                row = {**scores(y, lab), "decisions": dm.report["decisions"], "decisions_per_point": dm.report["decisions_per_point"]}
                if label == "d-means":
                    row["seconds_without_jev"] = time.perf_counter() - t0
                res.setdefault(label, []).append(row)
            log.info("%s seed %d done", name, seed)
        out["datasets"][name] = {
            "n": len(ds.texts),
            "k": ds.k,
            "about": ds.about,
            "runs": res,
            "mean": {m: _mean(r) for m, r in res.items()},
        }
    return out


STABLE = ("acc", "nmi", "ari", "decisions")


def verify(result: dict, reference: dict) -> list[str]:
    """Problems found: drift from the committed results, or d-means not beating every baseline."""
    errors = []
    for name, ref in reference["datasets"].items():
        got = result["datasets"].get(name)
        if got is None:
            errors.append(f"{name}: not run")
            continue
        for method, runs in ref["runs"].items():
            for i, (a, b) in enumerate(zip(runs, got["runs"].get(method, []))):
                for key in STABLE:
                    if key in a and abs(a[key] - b[key]) > 1e-6:
                        errors.append(f"{name}/{method}/seed{i}: {key} {b[key]:.6f} != committed {a[key]:.6f}")
        mean = got["mean"]
        for metric in ("acc", "nmi", "ari"):
            best = max(mean[m][metric] for m in BASELINES)
            if mean["d-means"][metric] <= best:
                errors.append(f"{name}: d-means {metric} {mean['d-means'][metric]:.4f} does not beat best baseline {best:.4f}")
        for i, run_ in enumerate(got["runs"]["d-means"]):
            best_seed = max(got["runs"][m][i]["acc"] for m in BASELINES)
            if run_["acc"] <= best_seed:
                errors.append(f"{name}/seed{i}: d-means acc {run_['acc']:.4f} does not beat best baseline {best_seed:.4f}")
    return errors


def table(result: dict) -> str:
    lines = []
    for name, d in result["datasets"].items():
        lines.append(f"\n{name}  (n={d['n']}, k={d['k']}, mean of {len(result['seeds'])} seeds)")
        lines.append(f"  {'method':38s} {'ACC':>7s} {'NMI':>7s} {'ARI':>7s} {'decisions/pt':>13s}")
        for m, v in d["mean"].items():
            dpp = f"{v['decisions_per_point']:.3f}" if "decisions_per_point" in v else "-"
            lines.append(f"  {m:38s} {v['acc']:7.3f} {v['nmi']:7.3f} {v['ari']:7.3f} {dpp:>13s}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--live", action="store_true", help="call Jev for answers missing from the cache")
    ap.add_argument("--write", action="store_true", help="save results/semantic.json and prune the answer cache")
    ap.add_argument("--verify", action="store_true", help="fail unless results match results/semantic.json and d-means wins")
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args(argv)
    _log.setup()
    jev = Jev(ANSWERS, offline=not a.live)
    try:
        result = run(a.datasets.split(","), [int(s) for s in a.seeds.split(",")], jev, ablations=not a.no_ablations)
    except CacheMiss as e:
        log.error("Jev answer %s is not in %s; run with --live to ask Jev", e, ANSWERS)
        return 2
    finally:
        if a.live:
            jev.save()
    result["jev_requests"] = jev.stats["requests"]
    result["jev_api_calls"] = jev.stats["api_calls"]
    print(table(result))
    if a.write:
        RESULTS.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        jev.save(keep=jev.used)
        log.info("wrote %s", OUT)
    if a.verify:
        errors = verify(result, json.loads(OUT.read_text()))
        for e in errors:
            log.error("verify: %s", e)
        if errors:
            return 1
        log.info("verify: results match %s; %d requests answered from %d cached Jev answers, %d API calls; d-means beats every baseline", OUT.name, jev.stats["requests"], len(jev.used), jev.stats["api_calls"])
        if jev.stats["api_calls"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
