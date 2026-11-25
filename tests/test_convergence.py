"""Convergence benchmark: the committed curves, the simulated decider and the SVG charts."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bench import chart, convergence as cv
from bench.report import ASSETS, charts
from dmeans.dmeans import assign_question, merge_question

NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def committed():
    return json.loads(cv.OUT.read_text())


def test_fastest_d_means_reaches_target_in_fewest_epochs(committed):
    assert committed["problems"], "no problem in results/convergence.json"
    assert cv.fastest_problems(committed) == list(committed["problems"])
    for p in committed["problems"].values():
        runs = [{"acc": {m: p["acc"][m][i] for m in cv.METHODS}, "epochs": p["epochs"], "passes": p["passes"], "decisions_per_point": [p["decisions_per_point"]]} for i in range(len(p["seeds"]))]
        again = cv.summarize(runs)  # the stored summary follows from the stored per-seed accuracies
        assert again["target"] == p["target"] and again["epochs_to_target"] == p["epochs_to_target"]
        assert all(again["mean"][m] == pytest.approx(p["mean"][m]) for m in cv.METHODS)
        reach = p["epochs_to_target"]
        assert all(reach[m] is None or reach["d-means"] < reach[m] for m in cv.BASELINES)
        assert p["mean"]["d-means"][0] >= p["target"]["acc"]  # already past it at the smallest budget


@pytest.fixture(scope="module")
def decider_case():
    prob = cv.topics(n=900)
    texts, y = prob["texts"], prob["labels"]
    rng = np.random.default_rng(7)
    by_topic = [np.flatnonzero(y == t) for t in range(prob["k"])]

    def card(t):
        return {"keywords": "", "examples": [texts[i] for i in rng.choice(by_topic[t], 4, replace=False)]}

    reqs, truth = [], []
    for _ in range(1500):
        a, b = rng.integers(prob["k"], size=2)
        reqs.append({"state": {"group_a": card(a), "group_b": card(b)}, "questions": {"same": merge_question("topic")}})
        truth.append(("noul", a == b))
    for _ in range(1500):
        i = int(rng.integers(len(texts)))
        opts = rng.choice(prob["k"], 4, replace=False)
        crit = {f"group {g}": card(g) for g in opts}
        reqs.append({"state": {"text": texts[i]}, "questions": {"group": assign_question("topic", crit)}})
        truth.append(("choice", f"group {y[i]}" if y[i] in opts else None))
    return prob, reqs, truth


def test_decider_is_wrong_at_its_stated_rate_and_repeatable(decider_case, committed):
    prob, reqs, truth = decider_case
    dec = cv.SimulatedDecider(prob["texts"], prob["labels"])
    ans = dec.ask(reqs)
    wrong = 0
    for a, (kind, t) in zip(ans, truth):
        if kind == "noul":
            wrong += (a["same"] > 0.5) != t
        elif t is not None:  # a Choice whose options miss the text's topic has no right answer
            wrong += max(a["group"], key=a["group"].get) != t
    graded = sum(1 for kind, t in truth if kind == "noul" or t is not None)
    assert abs(wrong / graded - cv.ERROR) <= 0.02
    assert abs(dec.stats["flipped"] / dec.stats["requests"] - cv.ERROR) <= 0.02
    assert cv.SimulatedDecider(prob["texts"], prob["labels"]).ask(reqs[::-1]) == ans[::-1]  # order-free
    for p in committed["problems"].values():
        if "decider_flip_rate" in p:
            assert abs(p["decider_flip_rate"] - cv.ERROR) <= 0.02


@pytest.mark.parametrize("theme", sorted(chart.THEMES))
def test_svg_has_four_lines_axes_and_legend(committed, theme):
    for name, p in committed["problems"].items():
        body = chart.svg(p, theme)
        root = ET.fromstring(body)  # well-formed XML
        assert root.tag == f"{NS}svg" and root.get("role") == "img"
        groups = {g.get("data-method"): g for g in root.iter(f"{NS}g") if g.get("data-method")}
        assert set(groups) == set(cv.METHODS)
        for m, g in groups.items():
            assert len(g.findall(f"{NS}path")) == 1
            assert len(g.findall(f"{NS}circle")) == len(p["mean"][m])
        assert {g.get("data-legend") for g in root.iter(f"{NS}g") if g.get("data-legend")} == set(cv.METHODS)
        texts = {t.text for t in root.iter(f"{NS}text")}
        assert {"Accuracy (%)", "Epochs: passes over the data (log scale)", chart.TITLE} <= texts
        assert chart.THEMES[theme]["ink"] in body
        committed_file = ASSETS / f"convergence-{name}-{theme}.svg"
        assert committed_file.read_text() == body
    assert charts(), "bench.report renders no chart"


def test_svg_themes_differ_only_in_color(committed):
    p = next(iter(committed["problems"].values()))
    light, dark = chart.svg(p, "light"), chart.svg(p, "dark")
    assert light != dark
    for key, value in chart.THEMES["light"].items():
        dark = dark.replace(chart.THEMES["dark"][key], value)
    assert dark == light
