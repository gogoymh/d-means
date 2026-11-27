"""Semantic clustering benchmark: k-means vs e-means vs o-means vs d-means on labeled texts.

    uv run python -m bench.semantic            # replay Clef's cached answers, print the table
    uv run python -m bench.semantic --verify   # also check it matches results/semantic-clef.json
    uv run python -m bench.semantic --live --write   # ask Clef for missing answers, save results
    uv run python -m bench.semantic --model typesafe/jev-1.14 --verify   # the same with Jev

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
from sklearn.cluster import KMeans

from dmeans import DMeans, _log
from dmeans.geometry import nearest
from dmeans.deciders import CacheMiss, Decisions
from dmeans.metrics import scores

from . import data, embed, official

log = _log.get("bench.semantic")
RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "cloudflare/clef"  # the main table's decision model
JEV = "typesafe/jev-1.14"  # the convergence charts' decision model
MODELS = {
    MODEL: "Clef",
    JEV: "Jev 1.14",
    "cloudflare/clef-flash": "Clef-flash",
    "google/gemini-decision-1": "Gemini Decision 1",
    "openai/gpt-6-luna-decisions": "Luna Decisions",
    "liquid/d1.2-ultra": "d1.2-ultra",
}
DATASETS = ["trec", "stackexchange", "ag-news", "bbc-news", "20-newsgroups", "banking77"]  # every loader in bench/data.py
SEEDS = [0, 1, 2]
BASELINES = ["k-means", "e-means", "o-means"]
# Each model records its own cost per decision from usage.cost when run with --live --write.


def slug(model: str) -> str:
    """"typesafe/jev-1.14" -> "jev", "cloudflare/clef-flash" -> "clef-flash"."""
    name = model.split("/")[-1]
    return name.split("-1.")[0] if name.startswith("jev") else name


def paths(model: str = MODEL) -> tuple[Path, Path]:
    """(answer cache, results file) of one decision model."""
    s = slug(model)
    return RESULTS / f"{s}-answers.jsonl.gz", RESULTS / f"semantic-{s}.json"


ANSWERS, OUT = paths()

RESTARTS = 10  # K-means baseline initializations: the same start the default d-means uses

ABLATIONS = {
    "d-means": {},
    "k-means start only (no decisions)": {"budget": 0.0},
    "point budget 15%": {"budget": 0.15},
    "decision only (no geometric prior)": {"fusion": "decision"},
    "merge start (v1 strategy)": {"start": "merge", "budget": 0.15},
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
    return "k-means start only (no decisions)" in d.get("runs", {})


def run_baseline(method: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "k-means":
        return KMeans(k, n_init=RESTARTS, random_state=seed).fit(x).labels_
    c, _ = {"e-means": official.emeans, "o-means": official.omeans}[method](x, k, nepochs=5, seed=seed)
    return nearest(x, c)[0]


def _mean(runs: list[dict]) -> dict:
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def run(datasets: list[str], seeds: list[int], jev: Decisions, ablations: bool = True) -> dict:
    out = {
        "jev_model": jev.model,
        "embedder": embed.MODEL,
        "seeds": seeds,
        "strategy": "kmeans",
        "kmeans_restarts": RESTARTS,
        "usd_per_decision": {},
        "datasets": {},
    }
    for name in datasets:
        calls0, cost0, used0 = jev.stats["api_calls"], jev.stats["cost_usd"], set(jev.used)
        ds = data.load(name)
        x = embed.embed(name, ds.texts)
        y = ds.labels
        res: dict[str, list[dict]] = {}
        for seed in seeds:
            for m in BASELINES:
                t0 = time.perf_counter()
                lab = run_baseline(m, x, ds.k, seed)
                res.setdefault(m, []).append({**scores(y, lab), "seconds": time.perf_counter() - t0})
            variants = ABLATIONS if ablations else {"d-means": {}}
            for label, kw in variants.items():
                dm = DMeans(ds.k, jev, about=ds.about, seed=seed, **kw)
                t0 = time.perf_counter()
                lab = dm.fit(x, ds.texts)
                row = {**scores(y, lab), "decisions": dm.report["decisions"], "decisions_per_point": dm.report["decisions_per_point"]}
                if label == "d-means":
                    row["seconds_without_jev"] = time.perf_counter() - t0
                res.setdefault(label, []).append(row)
            log.info("%s seed %d done", name, seed)
        calls, cost = jev.stats["api_calls"] - calls0, jev.stats["cost_usd"] - cost0
        log.info("%s: %d %s API calls cost $%.5f ($%.8f per call)", name, calls, jev.model, cost, cost / calls if calls else 0.0)
        if calls:
            out["usd_per_decision"][name] = round(cost / calls, 8)
        out["datasets"][name] = {
            "n": len(ds.texts),
            "k": ds.k,
            "about": ds.about,
            "runs": res,
            "mean": {m: _mean(r) for m, r in res.items()},
        }
        refused = len((jev.used - used0) & jev.refused)
        if refused:
            out["datasets"][name]["refusals"] = refused  # answered with neutral probabilities
            log.info("%s: %s refused %d requests, answered neutrally", name, jev.model, refused)
    return out


STABLE = ("acc", "nmi", "ari", "decisions")


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
    ap.add_argument("--write", action="store_true", help="save the results file and prune unused answers from the cache")
    ap.add_argument("--verify", action="store_true", help="fail unless results match the committed results file")
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args(argv)
    _log.setup()
    answers, out = paths(a.model)
    jev = Decisions(a.model, answers, offline=not a.live)
    try:
        names = a.datasets.split(",") if a.datasets else committed(out)
        seeds = [int(s) for s in a.seeds.split(",")]
        result = run(names, seeds, jev, ablations=not a.no_ablations and has_ablations(out))
    except CacheMiss as e:
        log.error("%s answer %s is not in %s; run with --live to ask it", a.model, e, answers)
        return 2
    finally:
        if a.live:
            jev.save()
    if out.exists():  # an offline replay keeps the cost rates measured live
        rates = json.loads(out.read_text()).get("usd_per_decision", {})
        result["usd_per_decision"] = {**rates, **result["usd_per_decision"]}
    result["jev_requests"] = jev.stats["requests"]
    result["jev_api_calls"] = jev.stats["api_calls"]
    result["jev_answers"] = len(jev.used)
    log.info("%s: %d requests, %d answers used, %d API calls, $%.5f, %d retries", a.model, jev.stats["requests"], len(jev.used), jev.stats["api_calls"], jev.stats["cost_usd"], jev.stats["retries"])
    print(table(result))
    if a.write:
        RESULTS.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
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
