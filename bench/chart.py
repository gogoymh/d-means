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


def comparison_svg(problems: dict) -> str:
    """Four method panels, sharing axes, from the recorded real-data curves."""
    width, height = 1080, 720
    colors = ["#265fb1", "#b5481e", "#0e7965", "#703caf"]
    descriptions = [
        "Centroids update after each full pass",
        "Centroids update after each mini-batch",
        "Mini-batches search nearby centroid cells",
        "Decisions merge groups and assign ambiguous texts",
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        '<title id="t">Four ways to cluster text</title>',
        '<desc id="d">K-means, E-means, O-means and D-means accuracy on TREC and StackExchange. Mean of three seeds. Recorded Jev 1.14 runs with consensus disabled; pass counts exclude model calls.</desc>',
        f'<style>text{{font-family:{FONT};fill:#526176;font-size:13px}}.heading{{fill:#172033;font-weight:700}}.tick{{font-size:12px}}</style>',
        '<rect width="1080" height="720" fill="#f6f8fc"/>',
        '<text x="60" y="46" class="heading" style="font-size:36px">Four ways to cluster text</text>',
        '<text x="60" y="84" style="font-size:18px">Clustering accuracy across training budgets on two text datasets</text>',
        '<path d="M60 118 H90" stroke="#172033" stroke-width="2.5"/><circle cx="75" cy="118" r="4" fill="#172033"/>',
        '<text x="100" y="123" style="font-size:15px">TREC (500 texts, k = 6)</text>',
        '<path d="M300 118 H330" stroke="#172033" stroke-width="2.5" stroke-dasharray="8 5"/><rect x="311" y="114" width="8" height="8" fill="#172033"/>',
        '<text x="340" y="123" style="font-size:15px">StackExchange (3,000 texts, k = 50)</text>',
    ]
    for index, method in enumerate(cv.METHODS):
        left = 113 + (index % 2) * 502
        top = 198 + (index // 2) * 253
        panel_width, panel_height = 422, 158
        color = colors[index]

        def sx(value):
            return left + (math.log2(value) + 6) / 10 * panel_width

        def sy(value):
            return top + (69 - value * 100) / 61 * panel_height

        out += [
            f'<text x="{left}" y="{top - 34}" class="heading" style="font-size:24px;fill:{color}">{method.capitalize()}</text>',
            f'<text x="{left}" y="{top - 8}">{descriptions[index]}</text>',
            f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="white"/>',
        ]
        for value in range(10, 70, 10):
            y = sy(value / 100)
            out.append(f'<path d="M{left} {_f(y)} H{left + panel_width}" stroke="#dce3ed"/>')
            out.append(f'<text x="{left - 8}" y="{_f(y + 4)}" text-anchor="end" class="tick">{value}</text>')
        for exponent in [-5, -3, -1, 1, 3, 4]:
            x = sx(2 ** exponent)
            out.append(f'<path d="M{_f(x)} {top} V{top + panel_height}" stroke="#dce3ed"/>')
            if index >= 2:
                out.append(f'<text x="{_f(x)}" y="{top + panel_height + 18}" text-anchor="middle" class="tick">{frac(2 ** exponent)}</text>')
        for dataset in ["trec", "stackexchange"]:
            p = problems[dataset]
            points = [(sx(x), sy(y)) for x, y in zip(p["passes"][method], p["mean"][method])]
            path = " ".join(f"{'M' if i == 0 else 'L'}{_f(x)},{_f(y)}" for i, (x, y) in enumerate(points))
            dash = ' stroke-dasharray="9 5"' if dataset == "stackexchange" else ""
            out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.7"{dash}/>')
            for x, y in points:
                if dataset == "trec":
                    out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="2.8" fill="{color}" stroke="white" stroke-width="0.6"/>')
                else:
                    out.append(f'<rect x="{_f(x - 2.5)}" y="{_f(y - 2.5)}" width="5" height="5" fill="{color}" stroke="white" stroke-width="0.6"/>')
    out += [
        '<text transform="translate(47 410) rotate(-90)" text-anchor="middle" style="font-size:16px;fill:#172033">Clustering accuracy (%)</text>',
        '<text x="540" y="654" text-anchor="middle" style="font-size:16px;fill:#172033">Effective passes over the data (log scale)</text>',
        '<text x="60" y="695">Mean of three seeds. D-means uses Jev 1.14. Pass counts exclude decision-model calls.</text>',
        '</svg>',
    ]
    return "\n".join(out) + "\n"


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
        f"Mean clustering accuracy against passes over the data. The best baseline ends at {target:.1f}% ({p['target']['method'].capitalize()}). "
        "Epochs needed to reach it: " + ", ".join(f"{m.capitalize()} {'never' if reach[m] is None else f'{reach[m]:.3g}'}" for m in LEGEND) + "."
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
            f'<circle cx="{lx + 10}" cy="86" r="4" fill="{col}"/><text x="{lx + 28}" y="90"{bold}>{m.capitalize()}</text></g>'
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
    o.append(f'<text x="{LEFT + 8}" y="{_f(ty - 7)}">Best baseline final: {target:.1f}% ({p["target"]["method"].capitalize()}, {frac(p["target"]["epochs"])} epochs)</text>')
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
        o.append(f'<text x="{_f(gx - 6 if right else gx + 6)}" y="{TOP + 14}" class="k"{anchor}>D-means reaches it at {reach["d-means"]:.3g} epochs</text>')
    ex, ey = pts["d-means"][-1]
    o.append(f'<text x="{_f(sx(ex) + 10)}" y="{_f(sy(ey) + 4)}" class="k n">{ey:.1f}%</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"
