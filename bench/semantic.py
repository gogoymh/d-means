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
from sklearn.metrics import roc_auc_score

from dmeans import DMeans, EMeans, OMeans, _log
from dmeans.geometry import nearest
from dmeans.deciders import CacheMiss, Jev
from dmeans.metrics import accuracy, scores

from . import data, embed

log = _log.get("bench.semantic")
RESULTS = Path(__file__).resolve().parent / "results"
ANSWERS = RESULTS / "jev-answers.jsonl.gz"
OUT = RESULTS / "semantic.json"
DATASETS = ["trec", "stackexchange", "ag-news", "bbc-news", "20-newsgroups", "banking77"]  # every loader in bench/data.py
SEEDS = [0, 1, 2]
BASELINES = ["k-means", "e-means", "o-means"]
# Cost per Jev decision measured on the live runs (OpenRouter usage.cost / API calls, Jev 1.14).
# trec: 447 calls $0.01175 and stackexchange: 3,684 calls $0.09766 (2026-11-22).
USD_PER_DECISION = {
    "trec": round(0.01175 / 447, 8),
    "stackexchange": round(0.09766 / 3684, 8),
}

ABLATIONS = {
    "d-means": {},
    "merge only (no point decisions)": {"budget": 0.0},
    "point budget 5%": {"budget": 0.05},
    "point budget 10%": {"budget": 0.10},
    "decision only (no geometric prior)": {"fusion": "decision"},
    "10 merge neighbours": {"neighbors": 10},
    "spectral merge": {"linkage": "spectral"},
}


def committed() -> list[str]:
    """Datasets in results/semantic.json, in DATASETS order: what replay and --verify cover by default."""
    got = json.loads(OUT.read_text())["datasets"] if OUT.exists() else {}
    return [d for d in DATASETS if d in got] or DATASETS


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


def diagnose(dm: DMeans, x: np.ndarray, y: np.ndarray) -> dict:
    """Where d-means gains or loses, stage by stage, judged against the true labels.

    purity: share of points whose micro-cluster's majority class is their class, the ceiling
    for any merge of these micro-clusters. noul_auc: how well Jev's p(same) ranks the asked
    pairs, truth being "same majority class". moved_right / moved_wrong: settled points that
    left a group of another majority class for one of their own, and the reverse.
    """
    ml = nearest(x, dm.micro_centroids)[0]
    major = np.array([np.bincount(y[ml == c]).argmax() if (ml == c).any() else -1 for c in range(len(dm.micro_centroids))])
    out = {"purity": accuracy(y, major[ml])}
    if len(dm.pairs):
        ii, jj = np.array(dm.pairs).T
        same = major[ii] == major[jj]
        out["pair_same_rate"] = float(same.mean())
        if 0 < same.sum() < len(same):
            out["noul_auc"] = float(roc_auc_score(same, dm.p_same))
    if len(dm.asked):
        gmajor = np.array([np.bincount(y[dm.merge_labels == g]).argmax() if (dm.merge_labels == g).any() else -1 for g in range(dm.k)])
        before, after, truth = gmajor[dm.merge_labels[dm.asked]], gmajor[dm.labels[dm.asked]], y[dm.asked]
        out["moved_right"] = int(((before != truth) & (after == truth)).sum())
        out["moved_wrong"] = int(((before == truth) & (after != truth)).sum())
    return out


def run(datasets: list[str], seeds: list[int], jev: Jev, ablations: bool = True) -> dict:
    out = {
        "jev_model": jev.model,
        "embedder": embed.MODEL,
        "seeds": seeds,
        "usd_per_decision": USD_PER_DECISION,
        "datasets": {},
    }
    for name in datasets:
        calls0, cost0 = jev.stats["api_calls"], jev.stats["cost_usd"]
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
                    row.update(diagnose(dm, x, y))
                res.setdefault(label, []).append(row)
            log.info("%s seed %d done", name, seed)
        calls, cost = jev.stats["api_calls"] - calls0, jev.stats["cost_usd"] - cost0
        log.info("%s: %d Jev API calls cost $%.5f ($%.8f per call)", name, calls, cost, cost / calls if calls else 0.0)
        out["datasets"][name] = {
            "n": len(ds.texts),
            "k": ds.k,
            "about": ds.about,
            "runs": res,
            "mean": {m: _mean(r) for m, r in res.items()},
        }
    return out


STABLE = ("acc", "nmi", "ari", "decisions", "purity", "noul_auc", "moved_right", "moved_wrong")


def verify(result: dict, reference: dict) -> list[str]:
    """Drift from the committed results. Losses are reported by losses(), not treated as errors."""
    errors = []
    for name, got in result["datasets"].items():
        ref = reference["datasets"].get(name)
        if ref is None:
            errors.append(f"{name}: not in the committed results")
            continue
        for method, runs in got["runs"].items():
            committed = ref["runs"].get(method)
            if committed is None or len(committed) != len(runs):
                errors.append(f"{name}/{method}: {len(runs)} runs, committed {len(committed or [])}")
                continue
            for i, (a, b) in enumerate(zip(committed, runs)):
                for key in STABLE:
                    if (key in a) != (key in b):
                        errors.append(f"{name}/{method}/seed{i}: {key} present in only one of run and committed")
                    elif key in a and abs(a[key] - b[key]) > 1e-6:
                        errors.append(f"{name}/{method}/seed{i}: {key} {b[key]:.6f} != committed {a[key]:.6f}")
    return errors


def losses(result: dict) -> dict[str, list[str]]:
    """Per dataset, the metrics (mean over seeds) on which d-means does not beat the best baseline."""
    out = {}
    for name, d in result["datasets"].items():
        mean = d["mean"]
        out[name] = [m for m in ("acc", "nmi", "ari") if mean["d-means"][m] <= max(mean[b][m] for b in BASELINES)]
    return out


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
    ap.add_argument("--datasets", default=None, help=f"comma-separated, from {','.join(DATASETS)} (default: the committed ones)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--live", action="store_true", help="call Jev for answers missing from the cache")
    ap.add_argument("--write", action="store_true", help="save results/semantic.json and prune the answer cache")
    ap.add_argument("--verify", action="store_true", help="fail unless results match results/semantic.json and d-means wins")
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args(argv)
    _log.setup()
    jev = Jev(ANSWERS, offline=not a.live)
    try:
        names = a.datasets.split(",") if a.datasets else committed()
        result = run(names, [int(s) for s in a.seeds.split(",")], jev, ablations=not a.no_ablations)
    except CacheMiss as e:
        log.error("Jev answer %s is not in %s; run with --live to ask Jev", e, ANSWERS)
        return 2
    finally:
        if a.live:
            jev.save()
    result["jev_requests"] = jev.stats["requests"]
    result["jev_api_calls"] = jev.stats["api_calls"]
    result["jev_answers"] = len(jev.used)
    log.info("Jev: %d requests, %d answers used, %d API calls, $%.5f, %d retries", jev.stats["requests"], len(jev.used), jev.stats["api_calls"], jev.stats["cost_usd"], jev.stats["retries"])
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
        lost = {n: m for n, m in losses(result).items() if m}
        log.info("verify: results match %s; %d requests answered from %d cached Jev answers, %d API calls", OUT.name, jev.stats["requests"], len(jev.used), jev.stats["api_calls"])
        log.info("verify: d-means beats the best baseline on every metric of %d of %d datasets; trails on %s", len(result["datasets"]) - len(lost), len(result["datasets"]), lost or "none")
        if jev.stats["api_calls"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
