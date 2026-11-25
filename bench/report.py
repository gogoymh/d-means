"""Render the README benchmark tables from results/*.json.

    uv run python -m bench.report          # rewrite the tables in README.md
    uv run python -m bench.report --check  # fail if README.md is out of date
"""

import argparse
import json
import re
import sys
from pathlib import Path

from .semantic import BASELINES, RESULTS

README = Path(__file__).resolve().parent.parent / "README.md"
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


def render(text: str) -> str:
    sem = json.loads((RESULTS / "semantic.json").read_text())
    blocks = {"semantic": semantic_table(sem), "ablation": ablation_table(sem)}
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
