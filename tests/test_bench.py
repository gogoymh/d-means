"""Benchmark bookkeeping, offline: drift checks, file names, and the README it renders."""

import copy
import json
from pathlib import Path

from bench import report, semantic

ROOT = Path(__file__).resolve().parent.parent


def _result(accs: dict[str, tuple[float, float]]) -> dict:
    """A fake results file: per dataset, (best baseline acc, d-means acc)."""
    out = {"seeds": [0], "usd_per_decision": {"a": 1e-5}, "jev_answers": 1234, "datasets": {}}
    for name, (base, dm) in accs.items():
        row = lambda a: {"acc": a, "nmi": a, "ari": a}  # noqa: E731
        runs = {"k-means": [row(base)], "e-means": [row(base - 0.01)], "o-means": [row(base - 0.02)], "d-means": [{**row(dm), "decisions": 100.0}]}
        out["datasets"][name] = {"n": 10, "k": 2, "runs": runs, "mean": {m: dict(r[0]) for m, r in runs.items()}}
    return out


def test_verify_flags_drift_but_not_losses():
    ref = _result({"a": (0.5, 0.6), "b": (0.7, 0.6)})
    assert semantic.verify(copy.deepcopy(ref), ref) == []
    assert semantic.losses(ref) == {"a": [], "b": ["acc", "nmi", "ari"]}
    drift = copy.deepcopy(ref)
    drift["datasets"]["a"]["runs"]["d-means"][0]["decisions"] = 90.0
    assert any("decisions" in e for e in semantic.verify(drift, ref))
    extra = _result({"a": (0.5, 0.6), "c": (0.5, 0.6)})
    assert semantic.verify(extra, ref) == ["c: not in the committed results"]


def test_summary_names_every_loss():
    res = _result({"a": (0.5, 0.6), "b": (0.7, 0.65)})
    text = report.summary(res)
    assert "on 1 of 2 datasets" in text and "It trails on b (ACC -5.0, NMI -5.0, ARI -5.0)" in text
    assert "1,234 decisions" in text
    table = report.semantic_table(res)
    assert "| a | 50.0 | 49.0 | 48.0 | **60.0** |" in table
    assert "| a | 10 | 2 | +10.0 | $0.001 |" in table
    assert "| b | **70.0** |" in table


def test_readme_explains_each_dataset_d_means_loses_on():
    res = json.loads((semantic.OUT).read_text())
    readme = (ROOT / "README.md").read_text()
    if not any(semantic.losses(res).values()):
        return
    section = readme.split("### Where D-means loses", 1)[1].split("\n## ", 1)[0]
    for name, lost in semantic.losses(res).items():
        if lost:
            assert f"**{report.NAMES[name]}**" in section, f"README gives no cause for {name}"



def test_models_table_bolds_the_best_and_counts_wins():
    clef = _result({"trec": (0.58, 0.62), "banking77": (0.63, 0.68)})
    gemini = _result({"trec": (0.58, 0.64), "banking77": (0.63, 0.60)})
    table = report.models_table({"cloudflare/clef": clef, "google/gemini-decision-1": gemini})
    assert "| Dataset | Best baseline | Clef | Gemini Decision 1 |" in table
    assert "| TREC | 58.0 | 62.0 | **64.0** |" in table
    assert "| Banking77 | 63.0 | **68.0** | 60.0 |" in table
    assert "beats the best baseline: Clef 2 of 2 and Gemini Decision 1 1 of 2." in table
    partial = report.models_table({"cloudflare/clef": clef, "google/gemini-decision-1": _result({"trec": (0.58, 0.64)})})
    assert "| Banking77 | 63.0 | **68.0** | - |" in partial
    assert "Gemini Decision 1 still needs the remaining datasets." in partial


def test_each_model_keeps_its_own_files():
    assert semantic.MODEL == "cloudflare/clef"
    assert [p.name for p in semantic.paths(semantic.MODEL)] == ["clef-answers.jsonl.gz", "semantic-clef.json"]
    assert [p.name for p in semantic.paths(semantic.JEV)] == ["jev-answers.jsonl.gz", "semantic-jev.json"]


def test_how_it_works_figures_match_their_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("steps", ROOT / "assets" / "steps.py")
    steps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(steps)
    for theme, colors in steps.THEMES.items():
        assert (ROOT / "assets" / f"step-select-{theme}.svg").read_text() == steps.select(colors)
        assert (ROOT / "assets" / f"step-ask-{theme}.svg").read_text() == steps.ask(colors)
