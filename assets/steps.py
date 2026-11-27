"""Draw the two "How it works" figures in README.md, light and dark, as self-contained SVGs.

    uv run python assets/steps.py

step-select-{light,dark}.svg: how a message's margin is measured and which 30% get a question.
step-ask-{light,dark}.svg: the question the decision model reads and the probabilities it returns.
The example messages and numbers are illustrative, as the README says next to them.
"""

import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The convergence charts' palette (bench/chart.py), plus a highlight for the message we follow.
THEMES = {
    "light": {"bg": "#ffffff", "panel": "#f6f8fa", "ink": "#1f2328", "ink2": "#59636e", "line": "#d1d9e0", "g1": "#257de0", "g2": "#f47230", "g3": "#00b894", "g1s": "#e6f0fc", "g2s": "#fdeee6", "g3s": "#e0f7f1", "hit": "#bf3989"},
    "dark": {"bg": "#0d1117", "panel": "#151b23", "ink": "#f0f6fc", "ink2": "#9198a1", "line": "#3d444d", "g1": "#338ced", "g2": "#e76624", "g3": "#00a888", "g1s": "#132338", "g2s": "#33200f", "g3s": "#0c2a24", "hit": "#db61a2"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def _svg(w: int, h: int, title: str, body: list[str], c: dict) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n'
        f"<style>text{{font-family:{FONT};fill:{c['ink']}}} .m{{fill:{c['ink2']}}} .mono{{font-family:{MONO}}} .b{{font-weight:700}}</style>\n"
        f'<rect width="{w}" height="{h}" rx="12" fill="{c["bg"]}" stroke="{c["line"]}"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def _t(x, y, s, size=15, cls="", fill=None, anchor="start"):
    f = f' style="fill:{fill}"' if fill else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}" text-anchor="{anchor}"{f}>{s}</text>'


def select(c: dict) -> str:
    """Two messages and their margins (left), 600 messages ranked by margin with the 30% cut (right)."""
    b = [_t(28, 40, "Which messages get a question?", 19, "b")]

    def example(y, pos, text, nums, verdict, color, near):
        out = [
            f'<rect x="32" y="{y - 6}" width="12" height="12" fill="{c["g1"]}"/>',
            f'<rect x="372" y="{y - 6}" width="12" height="12" fill="{c["g2"]}"/>',
            f'<line x1="{pos}" y1="{y}" x2="44" y2="{y}" stroke="{c["g1"]}" stroke-width="{2.5 if near == 1 else 1.5}" stroke-dasharray="{"0" if near == 1 else "4 4"}"/>',
            f'<line x1="{pos}" y1="{y}" x2="372" y2="{y}" stroke="{c["g2"]}" stroke-width="{2.5 if near == 2 else 1.5}" stroke-dasharray="{"0" if near == 2 else "4 4"}"/>',
            f'<circle cx="{pos}" cy="{y}" r="8" fill="{color}"/>',
            _t(32, y + 34, text, 15, "b", color if color == c["hit"] else None),
            _t(32, y + 56, nums, 14, "m mono"),
            _t(32, y + 78, verdict, 15, "b", color),
        ]
        return out

    b += example(84, 80, "“What is the fee for an international transfer?”", "d1 0.21   d2 1.84   margin 1.63", "Clear: keeps its K-means group", c["g1"], 1)
    b += example(214, 228, "“Why was I charged to send money?”", "d1 1.02   d2 1.10   margin 0.08", "Ambiguous: the model is asked", c["hit"], 2)

    # Ranked margins of 600 messages, binned into 60 bars.
    x0, w, base, top, n = 460, 400, 262, 70, 60
    rnd = random.Random(5)
    vals = sorted(rnd.random() ** 0.7 for _ in range(n))
    bw = w / n
    b.append(_t(x0, 40, "600 messages, smallest margin first", 15, "b"))
    for i, v in enumerate(vals):
        h = 8 + v * (base - top - 8)
        fill = c["hit"] if i < n * 0.3 else c["line"]
        b.append(f'<rect x="{x0 + i * bw + 0.7:.1f}" y="{base - h:.1f}" width="{bw - 1.4:.1f}" height="{h:.1f}" fill="{fill}"/>')
    cut = x0 + w * 0.3
    b.append(f'<line x1="{cut}" y1="{top - 8}" x2="{cut}" y2="{base + 4}" stroke="{c["ink"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    b.append(_t(cut + 8, top + 4, "budget = 30%", 13, "m"))
    b.append(f'<line x1="{x0}" y1="{base}" x2="{x0 + w}" y2="{base}" stroke="{c["ink2"]}"/>')
    b.append(_t(x0, base + 20, "ambiguous", 13, "m"))
    b.append(_t(x0 + w, base + 20, "clear", 13, "m", anchor="end"))
    b.append(_t(x0, base + 48, "First 180 (30%): asked", 15, "b", c["hit"]))
    b.append(_t(x0, base + 70, "Other 420 (70%): keep their K-means group", 15, "m"))
    return _svg(900, 350, "Margin = d2 − d1. The 30% of messages with the smallest margin are asked.", b, c)


def ask(c: dict) -> str:
    """The request D-means sends for one message (left) and the model's probabilities (right)."""
    b = [_t(28, 40, "What the decision model reads, and what it answers", 19, "b")]
    b.append(f'<rect x="28" y="60" width="470" height="262" rx="10" fill="{c["panel"]}" stroke="{c["line"]}"/>')
    b.append(_t(48, 90, "Message", 13, "m"))
    b.append(_t(48, 114, "“Why was I charged to send money?”", 17, "b", c["hit"]))
    b.append(_t(48, 146, "Question", 13, "m"))
    b.append(_t(48, 168, "Which group shares the customer's banking issue?", 15))
    b.append(_t(48, 198, "Candidate group cards, shuffled", 13, "m"))
    cards = [
        ("g2", "B", "pending, arrived", "“My transfer still hasn't arrived.”"),
        ("g3", "C", "card, lost, stolen", "“I lost my card yesterday.”"),
        ("g1", "A", "fee, charged", "“Why was I charged a transfer fee?”"),
    ]
    for i, (g, name, words, ex) in enumerate(cards):
        y = 208 + i * 36
        b.append(f'<rect x="48" y="{y}" width="430" height="30" rx="6" fill="{c[g + "s"]}"/>')
        b.append(f'<rect x="48" y="{y}" width="5" height="30" rx="2" fill="{c[g]}"/>')
        b.append(_t(62, y + 20, f"{name}", 14, "b", c[g]))
        b.append(_t(82, y + 20, f"{words}  ·  {ex}", 13))
    b.append(f'<path d="M 512 190 L 548 190" stroke="{c["ink2"]}" stroke-width="2"/><path d="M 546 184 L 556 190 L 546 196 Z" fill="{c["ink2"]}"/>')
    b.append(_t(572, 90, "Model probability", 13, "m"))
    rows = [("g1", "A  transfer fees", 0.85), ("g2", "B  transfer delays", 0.10), ("g3", "C  lost cards", 0.05)]
    for i, (g, name, p) in enumerate(rows):
        y = 118 + i * 66
        b.append(_t(572, y, name, 15, "b" if i == 0 else "", c[g]))
        b.append(f'<rect x="572" y="{y + 10}" width="240" height="18" rx="4" fill="{c["line"]}"/>')
        b.append(f'<rect x="572" y="{y + 10}" width="{max(3, 240 * p):.0f}" height="18" rx="4" fill="{c[g]}"/>')
        b.append(_t(872, y + 24, f"{p:.2f}", 15, "mono" + (" b" if i == 0 else ""), anchor="end"))
    return _svg(900, 350, "The model reads the message and the shuffled group cards, then returns a probability per group.", b, c)


def main() -> None:
    for theme, c in THEMES.items():
        (HERE / f"step-select-{theme}.svg").write_text(select(c))
        (HERE / f"step-ask-{theme}.svg").write_text(ask(c))
    print("wrote assets/step-select-*.svg and assets/step-ask-*.svg")


if __name__ == "__main__":
    main()
