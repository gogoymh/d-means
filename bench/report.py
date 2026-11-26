"""Render the README benchmark tables and the convergence charts from results/*.json.

    uv run python -m bench.report          # rewrite the tables in README.md and the SVGs in assets/
    uv run python -m bench.report --check  # fail if README.md or a chart is out of date
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

from . import chart
from .semantic import ANSWERS, BASELINES, MODEL, MODELS, RESULTS, losses, paths

README = Path(__file__).resolve().parent.parent / "README.md"
ASSETS = README.parent / "assets"
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
    model = res.get("jev_model", MODEL)
    answers = paths(model)[0]
    n = res.get("jev_answers")
    if n is None and model == MODEL and ANSWERS.exists():
        with gzip.open(ANSWERS, "rt") as f:
            n = sum(1 for _ in f)
    if n:
        out += f" Every d-means number comes from {MODELS.get(model, model)} answers through OpenRouter ({n:,} decisions, committed in [`bench/results/{answers.name}`](bench/results/{answers.name}))."
    return out


def semantic_table(res: dict) -> str:
    ds = list(res["datasets"])
    model = res.get("jev_model", MODEL)
    headers = [method.replace("-", "&#8209;") for method in METHODS]
    headers[METHODS.index("d-means")] = "**d&#8209;means**"
    rows = [
        "| Dataset | " + " | ".join(headers) + " |",
        "|:--|" + "--:|" * len(METHODS),
    ]
    run_details = [
        "| Dataset | Texts | Clusters | Lead (pp) | Cost/run |",
        "|:--|--:|--:|--:|--:|",
    ]
    for d in ds:
        r = res["datasets"][d]
        usd = _usd(res, d)
        cost = "-" if usd is None else f"${usd:.3f}"
        rows.append(f"| {_name(d)} | " + " | ".join(_cells(r["mean"], "acc")) + " |")
        run_details.append(
            f"| {_name(d)} | {r['n']:,} | {r['k']} | {_delta(r['mean'], 'acc')} | {cost} |"
        )
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
        + f"\n\nClustering accuracy (%), mean of seeds {seeds}. d-means uses {MODELS.get(model, model)}. "
        + summary(res)
        + "\n\n<details><summary>Run details, NMI and ARI</summary>\n\n"
        + "\n".join(run_details)
        + "\n\nLead is the accuracy difference from the best baseline, in percentage points.\n\n"
        + "\n".join(detail)
        + "\n\n</details>"
    )


def model_results() -> dict[str, dict]:
    """Every decision model with a committed results file, Jev (the main table) first."""
    out = {}
    for model in [MODEL] + sorted(set(MODELS) - {MODEL}):
        f = paths(model)[1]
        if f.exists():
            out[model] = json.loads(f.read_text())
    return out


def models_table(results: dict[str, dict]) -> str:
    """d-means ACC with each decision model, on every dataset any model has run; "-" where a model has not."""
    from .semantic import DATASETS

    ds = [d for d in DATASETS if any(d in r["datasets"] for r in results.values())]
    base = {d: next(r["datasets"][d]["mean"] for r in results.values() if d in r["datasets"]) for d in ds}
    acc = {m: {d: r["datasets"][d]["mean"]["d-means"]["acc"] for d in ds if d in r["datasets"]} for m, r in results.items()}
    floor = {d: max(base[d][b]["acc"] for b in BASELINES) for d in ds}
    top = {d: max([floor[d]] + [a[d] for a in acc.values() if d in a]) for d in ds}

    def cell(v: float, d: str) -> str:
        return f"**{_pct(v)}**" if v == top[d] else _pct(v)

    rows = [
        "| Decision model | " + " | ".join(_name(d) for d in ds) + " | wins | cost / decision | answers committed |",
        "|:--|" + "--:|" * (len(ds) + 3),
        "| none (best baseline) | " + " | ".join(cell(floor[d], d) for d in ds) + " | - | $0 | - |",
    ]
    for m, r in results.items():
        cells = [cell(acc[m][d], d) if d in acc[m] else "-" for d in ds]
        wins = sum(acc[m][d] > floor[d] for d in acc[m])
        usd = r.get("usd_per_decision") or {}
        rates = [usd[d] for d in acc[m] if isinstance(usd, dict) and d in usd]
        rate = "-" if not rates else (f"${min(rates) * 1e6:.0f}" if len(rates) == 1 else f"${min(rates) * 1e6:.0f} to {max(rates) * 1e6:.0f}") + " per 1M decisions"
        n = r.get("jev_answers", 0)
        rows.append(f"| {MODELS.get(m, m)} (`{m}`) | " + " | ".join(cells) + f" | {wins} of {len(acc[m])} | {rate} | [{n:,}](bench/results/{paths(m)[0].name}) |")
    seeds = ", ".join(map(str, results[MODEL]["seeds"]))
    others = [m for m in results if m != MODEL]
    replay = " ".join(f"`bench.semantic --model {m} --verify` replays {MODELS.get(m, m)} offline." for m in others)
    return (
        "\n".join(rows)
        + f"\n\nd-means ACC (%), mean of seeds {seeds}. Same embeddings, seeds, questions and defaults; only the model answering them changes. "
        + "*Wins* counts datasets where d-means beats the best baseline. *Cost* is OpenRouter's billed `usage.cost` per decision on the live run. "
        + replay
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


def diagnose_table(res: dict, model: str = MODEL) -> str:
    """The "Where d-means loses" section, for `model`'s run; empty until a live run has recorded stage diagnostics."""
    name = MODELS.get(model, model).split(" ")[0]
    if not any("purity" in r["mean"]["d-means"] for r in res["datasets"].values()):
        return ""
    rows = [
        f"| Dataset | best baseline | micro-cluster purity | geometric merge | {name} merge | after settle | Noul AUC | settle moves right / wrong |",
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
        f"\n\nACC (%) after each stage of a default run with {MODELS.get(model, model)}, mean of the same seeds. *Purity* is the ceiling for any merge of the 3k micro-clusters. "
        f"*Noul AUC* ranks {name}'s p(same) on the asked pairs against \"same majority class\" (0.5 is chance). "
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


def _decider(p: dict) -> str:
    d = chart.decider(p)
    return d if d != p["decider"] else "a " + d.replace("simulated,", "simulated decider,")


def reach_sentence(p: dict) -> tuple[str, str]:
    """(chart alt text, README sentence): whether d-means reaches the best baseline's final accuracy first, as measured."""
    reach = p["epochs_to_target"]
    d = reach["d-means"]
    base = min((reach[m], m) for m in BASELINES if reach[m] is not None)
    if d is None:
        best = max(p["mean"]["d-means"])
        return (
            "d-means never reaches the best baseline's final accuracy",
            f"d-means does not reach it at any budget: its best is {100 * best:.1f}%. {base[1]} gets there first, at {base[0]:.3g} epochs.",
        )
    when = f"at its first budget, {d:.3g} epochs" if d == p["passes"]["d-means"][0] else f"at {d:.3g} epochs"
    if d < base[0]:
        return (
            f"d-means clears the best baseline's final accuracy after {d:.3g} epochs",
            f"d-means passes it {when}, {base[0] / d:.0f}x sooner than the fastest baseline gets there ({base[1]}, {base[0]:.3g} epochs).",
        )
    return (
        f"d-means reaches the best baseline's final accuracy after {d:.3g} epochs, not first",
        f"d-means reaches it {when}; {base[1]} gets there first, at {base[0]:.3g} epochs.",
    )


def convergence_block(res: dict) -> str:
    parts = []
    for prob, p in res["problems"].items():
        img = f"assets/convergence-{prob}"
        target, reach = p["target"], p["epochs_to_target"]
        cols = [p["epochs"]["d-means"][0], 1, p["epochs"]["d-means"][-1]]
        head = "| Method | " + " | ".join(f"ACC at {chart.frac(t)} epoch{'s' if t > 1 else ''}" for t in cols) + f" | Epochs to reach {100 * target['acc']:.1f}% |"
        rows = [head, "|:--|" + "--:|" * (len(cols) + 1)]
        at = {m: dict(zip(p["epochs"][m], p["mean"][m])) for m in p["mean"]}
        fastest = min(v for v in reach.values() if v is not None)
        for m in at:
            cells = []
            for t in cols:
                best = max(a.get(t, 0.0) for a in at.values())
                cells.append("-" if t not in at[m] else (f"**{_pct(at[m][t])}**" if at[m][t] == best else _pct(at[m][t])))
            r = reach[m]
            cells.append("never" if r is None else (f"**{r:.3g}**" if r == fastest else f"{r:.3g}"))
            rows.append(f"| {'**d-means**' if m == 'd-means' else m} | " + " | ".join(cells) + " |")
        alt, verdict = reach_sentence(p)
        parts.append(
            f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{img}-dark.svg">\n'
            f'  <img alt="{chart.TITLE}: {alt}" src="{img}-light.svg" width="100%">\n</picture>\n\n'
            + "\n".join(rows)
            + f"\n\n{chart.name(p)}, mean of {len(p['acc']['d-means'])} seeds, decisions from {_decider(p)}. The best baseline ends at {100 * target['acc']:.1f}% "
            f"({target['method']}, {chart.frac(target['epochs'])} epochs). {verdict}"
        )
    return "\n\n".join(parts)


def charts() -> dict[Path, str]:
    """SVG files for every problem in results/convergence.json, light and dark."""
    path = RESULTS / "convergence.json"
    if not path.exists():
        return {}
    res = json.loads(path.read_text())
    return {ASSETS / f"convergence-{prob}-{theme}.svg": chart.svg(p, theme) for prob, p in res["problems"].items() for theme in chart.THEMES}


def render(text: str) -> str:
    sem = json.loads((RESULTS / "semantic.json").read_text())
    models = model_results()
    widest = max(models, key=lambda m: len(models[m]["datasets"]))  # use the broadest run for the main table and diagnostics
    blocks = {"semantic": semantic_table(models[widest]), "models": models_table(models), "ablation": ablation_table(sem), "diagnose": diagnose_table(models[widest], widest)}
    if (RESULTS / "mse.json").exists():
        blocks["mse"] = mse_table(json.loads((RESULTS / "mse.json").read_text()))
    if (RESULTS / "convergence.json").exists():
        blocks["convergence"] = convergence_block(json.loads((RESULTS / "convergence.json").read_text()))
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
    svgs = charts()
    if a.check:
        stale = (["README.md"] if old != new else []) + [f.name for f, body in svgs.items() if not f.exists() or f.read_text() != body]
        if stale:
            print(f"out of date: {', '.join(stale)}; run: uv run python -m bench.report", file=sys.stderr)
            return 1
        print(f"README.md tables and {len(svgs)} charts match bench/results")
        return 0
    README.write_text(new)
    for f, body in svgs.items():
        f.parent.mkdir(exist_ok=True)
        f.write_text(body)
    print(f"README.md tables and {len(svgs)} charts updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
