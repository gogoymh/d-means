"""Render the README benchmark tables and the convergence charts from results/*.json.

    uv run python -m bench.report          # rewrite the tables in README.md and the SVGs in assets/
    uv run python -m bench.report --check  # fail if README.md or a chart is out of date
"""

import argparse
import json
import re
import sys
from pathlib import Path

from . import chart
from .semantic import BASELINES, RESULTS

README = Path(__file__).resolve().parent.parent / "README.md"
ASSETS = README.parent / "assets"
NAMES = {"trec": "TREC", "stackexchange": "StackExchange"}


def _pct(v: float) -> str:
    return f"{100 * v:.1f}"


def semantic_table(res: dict) -> str:
    ds = list(res["datasets"])
    head = "| Method | " + " | ".join(f"{NAMES.get(d, d)} ACC | NMI | ARI" for d in ds) + " |"
    sep = "|:--|" + "--:|" * (3 * len(ds))
    rows = [head, sep]
    for m in BASELINES + ["d-means"]:
        cells = []
        for d in ds:
            mean = res["datasets"][d]["mean"]
            for metric in ("acc", "nmi", "ari"):
                v = _pct(mean[m][metric])
                best = max(mean[x][metric] for x in BASELINES + ["d-means"])
                cells.append(f"**{v}**" if mean[m][metric] == best else v)
        rows.append(f"| {'**d-means**' if m == 'd-means' else m} | " + " | ".join(cells) + " |")
    lines = "\n".join(rows)
    gain = []
    for d in ds:
        mean = res["datasets"][d]["mean"]
        best = max(BASELINES, key=lambda m: mean[m]["acc"])
        gain.append(f"{NAMES.get(d, d)} +{100 * (mean['d-means']['acc'] - mean[best]['acc']):.1f} ACC over {best}")
    return lines + "\n\n" + f"Mean of seeds {', '.join(map(str, res['seeds']))}. " + "; ".join(gain) + "."


def ablation_table(res: dict) -> str:
    ds = list(res["datasets"])
    rows = ["| Variant | " + " | ".join(f"{NAMES.get(d, d)} ACC | decisions / point" for d in ds) + " |", "|:--|" + "--:|" * (2 * len(ds))]
    order = ["geometric merge (no decisions)"] + [m for m in res["datasets"][ds[0]]["mean"] if m not in BASELINES and m != "geometric merge (no decisions)"]
    for m in order:
        cells = []
        for d in ds:
            v = res["datasets"][d]["mean"][m]
            cells += [_pct(v["acc"]), f"{v['decisions_per_point']:.2f}" if "decisions_per_point" in v else "0"]
        rows.append(f"| {'**d-means (default)**' if m == 'd-means' else m} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


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
            cells.append("never" if r is None else (f"**{r:.2g}**" if r == fastest else f"{r:.2g}"))
            rows.append(f"| {'**d-means**' if m == 'd-means' else m} | " + " | ".join(cells) + " |")
        base = min((reach[m], m) for m in BASELINES if reach[m] is not None)
        alt = f"{chart.TITLE}: d-means clears the best baseline's final accuracy after {reach['d-means']:.2g} epochs"
        parts.append(
            f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{img}-dark.svg">\n'
            f'  <img alt="{alt}" src="{img}-light.svg" width="100%">\n</picture>\n\n'
            + "\n".join(rows)
            + f"\n\n{chart.name(p)}, mean of {len(p['acc']['d-means'])} seeds. The best baseline ends at {100 * target['acc']:.1f}% "
            f"({target['method']}, {chart.frac(target['epochs'])} epochs). d-means passes it at its first budget, {reach['d-means']:.2g} epochs, "
            f"at least {base[0] / reach['d-means']:.0f}x sooner than the fastest baseline gets there ({base[1]}, {base[0]:.2g} epochs)."
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
    blocks = {"semantic": semantic_table(sem), "ablation": ablation_table(sem)}
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
