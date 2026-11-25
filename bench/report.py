"""Render the README benchmark tables from results/*.json.

    uv run python -m bench.report          # rewrite the tables in README.md
    uv run python -m bench.report --check  # fail if README.md is out of date
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

from .semantic import ANSWERS, BASELINES, RESULTS, losses

README = Path(__file__).resolve().parent.parent / "README.md"
NAMES = {
    "trec": "TREC",
    "stackexchange": "StackExchange",
    "ag-news": "AG News",
    "bbc-news": "BBC News",
    "20-newsgroups": "20 Newsgroups",
    "banking77": "Banking77",
}
METHODS = BASELINES + ["d-means"]
METRIC = {"acc": "ACC", "nmi": "NMI", "ari": "ARI"}


def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def _name(d: str) -> str:
    return NAMES.get(d, d)


def _cells(mean: dict, metric: str) -> list[str]:
    best = max(mean[m][metric] for m in METHODS)
    return [f"**{_pct(mean[m][metric])}**" if mean[m][metric] == best else _pct(mean[m][metric]) for m in METHODS]


def _delta(mean: dict, metric: str) -> str:
    return f"{100 * (mean['d-means'][metric] - max(mean[b][metric] for b in BASELINES)):+.1f}"


def _usd(res: dict, d: str) -> float | None:
    usd = res.get("usd_per_decision")
    rate = usd.get(d) if isinstance(usd, dict) else usd
    return None if rate is None else rate * res["datasets"][d]["mean"]["d-means"]["decisions"]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def summary(res: dict) -> str:
    """One generated paragraph: where d-means wins, where it trails, and what backs the numbers."""
    ds = list(res["datasets"])
    lost = losses(res)
    wins = [d for d in ds if not lost[d]]
    every_seed = [
        d for d in wins
        if all(r["acc"] > max(res["datasets"][d]["runs"][b][i]["acc"] for b in BASELINES) for i, r in enumerate(res["datasets"][d]["runs"]["d-means"]))
    ]
    out = f"d-means beats the best baseline on ACC, NMI and ARI on {len(wins)} of {len(ds)} datasets"
    out += f" ({_and([_name(d) for d in every_seed])} on every seed)." if every_seed else "."
    trails = []
    for d in ds:
        if lost[d]:
            mean = res["datasets"][d]["mean"]
            trails.append(f"{_name(d)} ({', '.join(METRIC[m] + ' ' + _delta(mean, m) for m in lost[d])})")
    if trails:
        out += f" It trails on {_and(trails)}; [why](#where-d-means-loses)."
    n = res.get("jev_answers")
    if n is None and ANSWERS.exists():
        with gzip.open(ANSWERS, "rt") as f:
            n = sum(1 for _ in f)
    if n:
        out += f" Every d-means number comes from real Jev 1.14 answers through OpenRouter ({n:,} decisions, committed in [`bench/results/jev-answers.jsonl.gz`](bench/results/jev-answers.jsonl.gz))."
    return out


def semantic_table(res: dict) -> str:
    ds = list(res["datasets"])
    rows = [
        "| Dataset | n | k | " + " | ".join("**d-means**" if m == "d-means" else m for m in METHODS) + " | vs best | Jev cost / run |",
        "|:--|--:|--:|" + "--:|" * (len(METHODS) + 2),
    ]
    for d in ds:
        r = res["datasets"][d]
        usd = _usd(res, d)
        cost = "-" if usd is None else f"${usd:.3f}"
        rows.append(f"| {_name(d)} | {r['n']:,} | {r['k']} | " + " | ".join(_cells(r["mean"], "acc")) + f" | {_delta(r['mean'], 'acc')} | {cost} |")
    detail = [
        "| Dataset | " + " | ".join(f"{m} NMI" for m in METHODS) + " | " + " | ".join(f"{m} ARI" for m in METHODS) + " |",
        "|:--|" + "--:|" * (2 * len(METHODS)),
    ]
    for d in ds:
        mean = res["datasets"][d]["mean"]
        detail.append(f"| {_name(d)} | " + " | ".join(_cells(mean, "nmi") + _cells(mean, "ari")) + " |")
    seeds = ", ".join(map(str, res["seeds"]))
    return (
        "\n".join(rows)
        + f"\n\nClustering accuracy (%), mean of seeds {seeds}. *vs best* is d-means minus the best baseline. "
        + summary(res)
        + "\n\n<details><summary>NMI and ARI</summary>\n\n"
        + "\n".join(detail)
        + "\n\n</details>"
    )


def ablation_table(res: dict) -> str:
    ds = list(res["datasets"])
    rows = [
        "| Variant | " + " | ".join(_name(d) for d in ds) + " | decisions / point |",
        "|:--|" + "--:|" * (len(ds) + 1),
        "| best baseline | " + " | ".join(_pct(max(res["datasets"][d]["mean"][b]["acc"] for b in BASELINES)) for d in ds) + " | 0 |",
    ]
    order = ["geometric merge (no decisions)"] + [m for m in res["datasets"][ds[0]]["mean"] if m not in BASELINES and m != "geometric merge (no decisions)"]
    for m in order:
        means = [res["datasets"][d]["mean"][m] for d in ds]
        dpp = [v.get("decisions_per_point", 0.0) for v in means]
        span = "0" if not any(dpp) else f"{min(dpp):.2f} to {max(dpp):.2f}"
        rows.append(f"| {'**d-means (default)**' if m == 'd-means' else m} | " + " | ".join(_pct(v["acc"]) for v in means) + f" | {span} |")
    return "\n".join(rows) + "\n\nACC (%), mean of the same seeds."


def diagnose_table(res: dict) -> str:
    """The "Where d-means loses" section; empty until a live run has recorded stage diagnostics."""
    if not any("purity" in r["mean"]["d-means"] for r in res["datasets"].values()):
        return ""
    rows = [
        "| Dataset | best baseline | micro-cluster purity | geometric merge | Jev merge | after settle | Noul AUC | settle moves right / wrong |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for d, r in res["datasets"].items():
        mean = r["mean"]
        dm = mean["d-means"]
        auc = f"{dm['noul_auc']:.2f}" if "noul_auc" in dm else "-"
        moves = f"{dm['moved_right']:.0f} / {dm['moved_wrong']:.0f}" if "moved_right" in dm else "-"
        cells = [
            _pct(max(mean[b]["acc"] for b in BASELINES)),
            _pct(dm["purity"]) if "purity" in dm else "-",
            _pct(mean["geometric merge (no decisions)"]["acc"]) if "geometric merge (no decisions)" in mean else "-",
            _pct(mean["merge only (no point decisions)"]["acc"]) if "merge only (no point decisions)" in mean else "-",
            _pct(dm["acc"]),
            auc,
            moves,
        ]
        rows.append(f"| {_name(d)} | " + " | ".join(cells) + " |")
    return "### Where d-means loses\n\n" + "\n".join(rows) + (
        "\n\nACC (%) after each stage of a default run, mean of the same seeds. *Purity* is the ceiling for any merge of the 3k micro-clusters. "
        "*Noul AUC* ranks Jev's p(same) on the asked pairs against \"same majority class\" (0.5 is chance). "
        "*Settle moves* count borderline points that left a group of another class for their own, and the reverse, per run."
    )


def mse_table(res: dict) -> str:
    rows = ["| Problem | Epochs | e-means MSE vs Lloyd@25 | o-means MSE vs Lloyd@25 | o-means distances vs e-means | Lloyd@25 time | e-means time | o-means time |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for name, r in res.items():
        lloyd_s = r["Lloyd"][-1]["seconds"]
        for e, o in zip(r["e-means"], r["o-means"]):
            rows.append(
                f"| {name} | {e['epochs']} | {100 * e['mse_vs_lloyd25']:+.2f}% | {100 * o['mse_vs_lloyd25']:+.2f}% | "
                f"{o['distance_evals_vs_emeans']:.2f}x | {lloyd_s:.1f}s | {e['seconds']:.1f}s | {o['seconds']:.1f}s |"
            )
    return "\n".join(rows)


def render(text: str) -> str:
    sem = json.loads((RESULTS / "semantic.json").read_text())
    blocks = {"semantic": semantic_table(sem), "ablation": ablation_table(sem), "diagnose": diagnose_table(sem)}
    if (RESULTS / "mse.json").exists():
        blocks["mse"] = mse_table(json.loads((RESULTS / "mse.json").read_text()))
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- bench:{name} -->\n).*?(<!-- /bench:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md has no bench:{name} block")
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    old = README.read_text()
    new = render(old)
    if a.check:
        if old != new:
            print("README.md benchmark tables are out of date; run: uv run python -m bench.report", file=sys.stderr)
            return 1
        print("README.md benchmark tables match bench/results")
        return 0
    README.write_text(new)
    print("README.md tables updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
