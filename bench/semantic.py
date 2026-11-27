"""Semantic clustering benchmark: k-means vs e-means vs o-means vs d-means on labeled texts.

    uv run python -m bench.semantic            # replay Jev's cached answers, print the table
    uv run python -m bench.semantic --verify   # also check it matches results/semantic.json
    uv run python -m bench.semantic --live --write   # ask Jev for missing answers, save results
    uv run python -m bench.semantic --model cloudflare/clef --verify   # the same with Clef
    uv run python -m bench.semantic --model cloudflare/clef --consensus --verify

Replay is offline and exact: d-means sends the same requests, and the committed cache returns
the answers the model gave when the benchmark was run live. A request missing from the cache
fails the replay instead of being guessed. Each model keeps its own answers and results file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from dmeans import DMeans, EMeans, OMeans, _log
from dmeans.geometry import nearest
from dmeans.deciders import CacheMiss, Decisions
from dmeans.metrics import accuracy, scores

from . import data, embed

log = _log.get("bench.semantic")
RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "typesafe/jev-1.14"  # default for the legacy semantic benchmark
MODELS = {
    MODEL: "Jev 1.14",
    "cloudflare/clef": "Clef",
    "cloudflare/clef-flash": "Clef-flash",
    "google/gemini-decision-1": "Gemini Decision 1",
    "openai/gpt-6-luna-decisions": "Luna Decisions",
    "liquid/d1.2-ultra": "d1.2-ultra",
}
DATASETS = ["trec", "stackexchange", "ag-news", "bbc-news", "20-newsgroups", "banking77"]  # every loader in bench/data.py
SEEDS = [0, 1, 2]
BASELINES = ["k-means", "e-means", "o-means"]
# Cost per Jev decision measured on its live runs (OpenRouter usage.cost / API calls, Jev 1.14).
# trec: 447 calls $0.01175 and stackexchange: 3,684 calls $0.09766 (2026-11-22).
# Other models record their own rate from usage.cost when run with --live --write.
USD_PER_DECISION = {
    "trec": round(0.01175 / 447, 8),
    "stackexchange": round(0.09766 / 3684, 8),
}


def slug(model: str) -> str:
    """"typesafe/jev-1.14" -> "jev", "cloudflare/clef-flash" -> "clef-flash"."""
    name = model.split("/")[-1]
    return name.split("-1.")[0] if name.startswith("jev") else name


def paths(model: str = MODEL) -> tuple[Path, Path]:
    """(answer cache, results file) of one decision model; Jev keeps the original names."""
    s = slug(model)
    return RESULTS / f"{s}-answers.jsonl.gz", RESULTS / ("semantic.json" if model == MODEL else f"semantic-{s}.json")


ANSWERS, OUT = paths()

RESTARTS = 10  # K-means baseline initializations: the same start the default d-means uses

# The legacy Jev and consensus results were recorded with the v1 merge strategy; replay them with it.
LEGACY = {"start": "merge", "budget": 0.15}

ABLATIONS = {
    "d-means": {},
    "k-means start only (no decisions)": {"budget": 0.0},
    "point budget 15%": {"budget": 0.15},
    "decision only (no geometric prior)": {"fusion": "decision"},
    "merge start (v1 strategy)": {"start": "merge", "budget": 0.15},
}

LEGACY_ABLATIONS = {
    "d-means": {},
    "merge only (no point decisions)": {"budget": 0.0},
    "point budget 5%": {"budget": 0.05},
    "point budget 10%": {"budget": 0.10},
    "decision only (no geometric prior)": {"fusion": "decision"},
    "10 merge neighbours": {"neighbors": 10},
    "spectral merge": {"linkage": "spectral"},
}


def committed(out: Path = OUT) -> list[str]:
    """Datasets in the results file, in DATASETS order: what replay and --verify cover by default."""
    got = json.loads(out.read_text())["datasets"] if out.exists() else {}
    return [d for d in DATASETS if d in got] or DATASETS


def has_ablations(out: Path) -> bool:
    """Whether the committed run includes the ablation variants (replay must ask the same questions)."""
    if not out.exists():
        return True
    d = next(iter(json.loads(out.read_text())["datasets"].values()), {})
    return bool({"merge only (no point decisions)", "k-means start only (no decisions)"} & set(d.get("runs", {})))


def geometric_merge(x: np.ndarray, k: int, seed: int, over: int = 3) -> np.ndarray:
    """d-means without decisions: same micro-clusters, merged by average-linkage on cosine."""
    k0 = int(min(over * k, max(k, len(x) // 4)))
    mc, _ = EMeans(k0, nepochs=5, seed=seed).fit(x)
    ml, _ = nearest(x, mc)
    group = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(mc)
    return group[ml]


def run_baseline(method: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "k-means":
        return KMeans(k, n_init=RESTARTS, random_state=seed).fit(x).labels_
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
        before, after, truth = gmajor[dm.merge_labels[dm.asked]], gmajor[dm.decision_labels[dm.asked]], y[dm.asked]
        out["moved_right"] = int(((before != truth) & (after == truth)).sum())
        out["moved_wrong"] = int(((before == truth) & (after != truth)).sum())
    return out


def run(datasets: list[str], seeds: list[int], jev: Decisions, ablations: bool = True, consensus: bool = False, legacy: bool = False, refine: bool = False) -> dict:
    out = {
        "jev_model": jev.model,
        "embedder": embed.MODEL,
        "seeds": seeds,
        "strategy": "merge" if legacy else "kmeans",
        "kmeans_restarts": RESTARTS,
        "usd_per_decision": dict(USD_PER_DECISION) if jev.model == MODEL else {},
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
            if refine:
                lab = KMeans(ds.k, n_init=8, random_state=seed).fit_predict(x)
                res.setdefault("k-means (8 starts)", []).append(scores(y, lab))
            if ablations and legacy:
                lab = geometric_merge(x, ds.k, seed)
                res.setdefault("geometric merge (no decisions)", []).append(scores(y, lab))
            variants = (LEGACY_ABLATIONS if legacy else ABLATIONS) if ablations else {"d-means": {}}
            for label, kw in variants.items():
                kw = {**(LEGACY if legacy else {}), **kw}
                dm = DMeans(ds.k, jev, about=ds.about, seed=seed, consensus=consensus, refine=refine, **kw)
                t0 = time.perf_counter()
                lab = dm.fit(x, ds.texts)
                row = {**scores(y, lab), "decisions": dm.report["decisions"], "decisions_per_point": dm.report["decisions_per_point"]}
                if label == "d-means":
                    row["seconds_without_jev"] = time.perf_counter() - t0
                    if refine:
                        row["refinement"] = dm.refinement
                    if consensus:
                        row["consensus_runs"] = int(dm.partition_source == "consensus")
                    if dm.partition_source == "decision" and dm.start == "merge":
                        row.update(diagnose(dm, x, y))
                res.setdefault(label, []).append(row)
            log.info("%s seed %d done", name, seed)
        calls, cost = jev.stats["api_calls"] - calls0, jev.stats["cost_usd"] - cost0
        log.info("%s: %d %s API calls cost $%.5f ($%.8f per call)", name, calls, jev.model, cost, cost / calls if calls else 0.0)
        if calls and jev.model != MODEL:
            out["usd_per_decision"][name] = round(cost / calls, 8)
        out["datasets"][name] = {
            "n": len(ds.texts),
            "k": ds.k,
            "about": ds.about,
            "runs": res,
            "mean": {m: _mean(r) for m, r in res.items()},
        }
    return out


STABLE = ("acc", "nmi", "ari", "decisions", "purity", "noul_auc", "moved_right", "moved_wrong", "consensus_runs")


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
    ap.add_argument("--model", default=MODEL, help=f"decision model on OpenRouter's Decisions API, e.g. {', '.join(MODELS)}")
    ap.add_argument("--datasets", default=None, help=f"comma-separated, from {','.join(DATASETS)} (default: the committed ones)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--live", action="store_true", help="ask the model for answers missing from the cache")
    ap.add_argument("--write", action="store_true", help="save the results file and prune the legacy answer cache")
    ap.add_argument("--verify", action="store_true", help="fail unless results match the committed results file")
    strategy = ap.add_mutually_exclusive_group()
    strategy.add_argument("--refined", action="store_true", help="evaluate the current decision-guided refinement")
    strategy.add_argument("--consensus", action="store_true", help="benchmark the adaptive embedding and lexical consensus strategy")
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args(argv)
    _log.setup()
    answers, legacy_out = paths(a.model)
    out = legacy_out
    if a.consensus:
        out = RESULTS / f"semantic-consensus-{slug(a.model)}.json"
    if a.refined:
        out = RESULTS / f"semantic-refined-{slug(a.model)}.json"
    jev = Decisions(a.model, answers, offline=not a.live)
    try:
        names = a.datasets.split(",") if a.datasets else committed(out)
        seeds = [int(s) for s in a.seeds.split(",")]
        # Jev and consensus results were recorded with the v1 merge strategy.
        legacy = a.consensus or a.model == MODEL
        if a.consensus or a.refined:
            with threadpool_limits(4):
                result = run(names, seeds, jev, ablations=False, consensus=a.consensus, legacy=a.consensus, refine=a.refined)
        else:
            result = run(names, seeds, jev, ablations=not a.no_ablations and has_ablations(out), legacy=legacy)
    except CacheMiss as e:
        log.error("%s answer %s is not in %s; run with --live to ask it", a.model, e, answers)
        return 2
    finally:
        if a.live:
            jev.save()
    if a.model != MODEL:
        rate_file = out if out.exists() else legacy_out
        if rate_file.exists():
            rates = json.loads(rate_file.read_text()).get("usd_per_decision", {})
            result["usd_per_decision"] = {**rates, **result["usd_per_decision"]}
    result["jev_requests"] = jev.stats["requests"]
    result["jev_api_calls"] = jev.stats["api_calls"]
    result["jev_answers"] = len(jev.used)
    log.info("%s: %d requests, %d answers used, %d API calls, $%.5f, %d retries", a.model, jev.stats["requests"], len(jev.used), jev.stats["api_calls"], jev.stats["cost_usd"], jev.stats["retries"])
    print(table(result))
    if a.write:
        RESULTS.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
        if not (a.consensus or a.refined):
            jev.save(keep=jev.used)
        log.info("wrote %s", out)
    if a.verify:
        if not out.exists():
            log.error("verify: %s does not exist; run with --live --write first", out)
            return 1
        errors = verify(result, json.loads(out.read_text()))
        for e in errors:
            log.error("verify: %s", e)
        if errors:
            return 1
        lost = {n: m for n, m in losses(result).items() if m}
        log.info("verify: results match %s; %d requests answered from %d cached %s answers, %d API calls", out.name, jev.stats["requests"], len(jev.used), a.model, jev.stats["api_calls"])
        log.info("verify: d-means beats the best baseline on every metric of %d of %d datasets; trails on %s", len(result["datasets"]) - len(lost), len(result["datasets"]), lost or "none")
        if jev.stats["api_calls"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
