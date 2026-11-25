"""Convergence chart as a self-contained SVG, one light and one dark version (no plotting library).

`svg(problem, theme)` draws mean accuracy against passes over the data (log scale) for every
method in results/convergence.json, with the target line every method is racing to. Output is
deterministic, so bench.report can check the committed files against the results.
"""

import math
from html import escape

from . import convergence as cv

W, H = 800, 484
LEFT, RIGHT, TOP, BOTTOM = 56, 72, 144, 420
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif"
# Baselines take the first three slots of a validated categorical palette (CVD-safe in both modes).
# d-means wears the ink color, the black of the brand, with a thicker line.
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "grid": "#eaeef2", "axis": "#d1d9e0", "k-means": "#257de0", "e-means": "#f47230", "o-means": "#00b894"},
    "dark": {"surface": "#0d1117", "ink": "#f0f6fc", "ink2": "#9198a1", "grid": "#21262d", "axis": "#3d444d", "k-means": "#338ced", "e-means": "#e76624", "o-means": "#00a888"},
}
LEGEND = ["d-means", "k-means", "e-means", "o-means"]
TITLE = "Clustering accuracy vs. epochs"


def frac(v: float) -> str:
    """1/32, 1/2, 1, 16: how an epoch budget reads on the axis and in the README."""
    return f"1/{round(1 / v)}" if v < 1 else f"{v:g}"


def _f(v: float) -> str:
    return f"{v:.1f}"


NAMES = {"synthetic": "Synthetic topics", "trec": "TREC", "stackexchange": "StackExchange", "ag-news": "AG News", "bbc-news": "BBC News", "20-newsgroups": "20 Newsgroups", "banking77": "Banking77"}


def decider(p: dict) -> str:
    return "Jev 1.14" if p["decider"] == "typesafe/jev-1.14" else p["decider"]


def name(p: dict) -> str:
    return NAMES.get(p["source"], p["source"])


def svg(p: dict, theme: str = "light") -> str:
    c = THEMES[theme]
    pts = {m: [(x, 100 * v) for x, v in zip(p["passes"][m], p["mean"][m])] for m in cv.METHODS}
    target = 100 * p["target"]["acc"]
    allx = [x for s in pts.values() for x, _ in s]
    ally = [y for s in pts.values() for _, y in s] + [target]
    lx0, lx1 = math.floor(math.log2(min(allx))), math.ceil(math.log2(max(allx)))
    y0, y1 = 10 * math.floor((min(ally) - 3) / 10), 10 * math.ceil((max(ally) + 3) / 10)

    def sx(v: float) -> float:
        return LEFT + (math.log2(v) - lx0) / (lx1 - lx0) * (W - LEFT - RIGHT)

    def sy(v: float) -> float:
        return BOTTOM - (v - y0) / (y1 - y0) * (BOTTOM - TOP)

    seeds = len(p["acc"]["d-means"])
    reach = p["epochs_to_target"]
    sub = f"{name(p)}, n={p['n']:,}, k={p['k']}. Mean of {seeds} seeds. Decisions: {decider(p)}, {p['decisions_per_point']:.2f} per point."
    desc = (
        f"Mean clustering accuracy against passes over the data. The best baseline ends at {target:.1f}% ({p['target']['method']}). "
        "Epochs needed to reach it: " + ", ".join(f"{m} {'never' if reach[m] is None else f'{reach[m]:.3g}'}" for m in LEGEND) + "."
    )
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">',
        f'<title id="t">{TITLE}</title><desc id="d">{escape(desc)}</desc>',
        f"<style>text{{font-family:{FONT};fill:{c['ink2']};font-size:12px}}.h{{fill:{c['ink']};font-size:17px;font-weight:600}}"
        f".n{{font-variant-numeric:tabular-nums}}.k{{fill:{c['ink']};font-weight:600}}</style>",
        f'<text class="h" x="{LEFT}" y="32">{TITLE}</text>',
        f'<text x="{LEFT}" y="54">{escape(sub)}</text>',
    ]
    # legend: a short line key with its dot, then the name in text ink (identity is never color alone)
    lx = LEFT
    for m in LEGEND:
        col, wid = (c["ink"], 3) if m == "d-means" else (c[m], 2)
        bold = ' class="k"' if m == "d-means" else ""
        o.append(
            f'<g data-legend="{m}"><line x1="{lx}" y1="86" x2="{lx + 20}" y2="86" stroke="{col}" stroke-width="{wid}" stroke-linecap="round"/>'
            f'<circle cx="{lx + 10}" cy="86" r="4" fill="{col}"/><text x="{lx + 28}" y="90"{bold}>{m}</text></g>'
        )
        lx += 28 + 7.2 * len(m) + 26
    # grid and axes
    o.append(f'<text x="{LEFT}" y="{TOP - 22}" class="k">Accuracy (%)</text>')
    for v in range(y0, y1 + 1, 10):
        o.append(f'<line x1="{LEFT}" y1="{_f(sy(v))}" x2="{W - RIGHT}" y2="{_f(sy(v))}" stroke="{c["grid"] if v > y0 else c["axis"]}" stroke-width="1"/>')
        o.append(f'<text x="{LEFT - 10}" y="{_f(sy(v) + 4)}" text-anchor="end" class="n">{v}</text>')
    for e in range(lx0, lx1 + 1):
        o.append(f'<text x="{_f(sx(2.0**e))}" y="{BOTTOM + 20}" text-anchor="middle" class="n">{frac(2.0**e)}</text>')
    o.append(f'<text x="{_f((LEFT + W - RIGHT) / 2)}" y="{BOTTOM + 48}" text-anchor="middle" class="k">Epochs: passes over the data (log scale)</text>')
    # the target every method races to: the best baseline's final accuracy
    ty = sy(target)
    o.append(f'<line x1="{LEFT}" y1="{_f(ty)}" x2="{W - RIGHT}" y2="{_f(ty)}" stroke="{c["ink2"]}" stroke-width="1" stroke-opacity="0.55"/>')
    o.append(f'<text x="{LEFT + 8}" y="{_f(ty - 7)}">Best baseline final: {target:.1f}% ({p["target"]["method"]}, {frac(p["target"]["epochs"])} epochs)</text>')
    # series: baselines first, d-means on top; dots carry a ring in the surface color
    for m in cv.BASELINES + ["d-means"]:
        col, wid, r = (c["ink"], 3, 4.5) if m == "d-means" else (c[m], 2, 3.5)
        d = " ".join(f"{'M' if i == 0 else 'L'}{_f(sx(x))},{_f(sy(y))}" for i, (x, y) in enumerate(pts[m]))
        o.append(f'<g data-method="{m}"><path d="{d}" fill="none" stroke="{col}" stroke-width="{wid}" stroke-linejoin="round" stroke-linecap="round"/>')
        o += [f'<circle cx="{_f(sx(x))}" cy="{_f(sy(y))}" r="{r}" fill="{col}" stroke="{c["surface"]}" stroke-width="2"/>' for x, y in pts[m]]
        o.append("</g>")
    # direct labels on the story only: where d-means reaches the target (a guide from the top), and where it ends
    if reach["d-means"] is not None:
        fx = reach["d-means"]
        fy = dict(pts["d-means"])[fx]
        gx, right = sx(fx), sx(fx) > (LEFT + W - RIGHT) / 2
        o.append(f'<line x1="{_f(gx)}" y1="{TOP + 6}" x2="{_f(gx)}" y2="{_f(sy(fy) - 8)}" stroke="{c["ink"]}" stroke-width="1" stroke-dasharray="3 3"/>')
        anchor = ' text-anchor="end"' if right else ""
        o.append(f'<text x="{_f(gx - 6 if right else gx + 6)}" y="{TOP + 14}" class="k"{anchor}>d-means reaches it at {reach["d-means"]:.3g} epochs</text>')
    ex, ey = pts["d-means"][-1]
    o.append(f'<text x="{_f(sx(ex) + 10)}" y="{_f(sy(ey) + 4)}" class="k n">{ey:.1f}%</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"
