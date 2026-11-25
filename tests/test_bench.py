"""Benchmark bookkeeping, offline: stage diagnostics, drift checks, and the README it renders."""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bench import report, semantic
from dmeans import DMeans

ROOT = Path(__file__).resolve().parent.parent


class Perfect:
    """A decision model that knows the classes: answers from the class word in each text."""

    def ask(self, reqs):
        def cls(t):
            return t.split()[0]

        def card_cls(card):
            words = [cls(t) for t in card["examples"]]
            return max(set(words), key=words.count)

        out = []
        for r in reqs:
            q = next(iter(r["questions"].values()))
            if q["type"] == "noul":
                same = card_cls(r["state"]["group_a"]) == card_cls(r["state"]["group_b"])
                out.append({"same": 0.95 if same else 0.05})
            else:
                truth = cls(r["state"]["text"])
                out.append({"group": {n: float(card_cls(c) == truth) for n, c in q["criteria"].items()}})
        return out


def test_diagnose_reads_each_stage():
    rng = np.random.default_rng(3)
    modes = rng.normal(size=(6, 8)).astype(np.float32) * 3
    names = ["alpha", "beta", "gamma"]
    x, texts, y = [], [], []
    for m in range(6):
        for i in range(40):
            x.append(modes[m] + rng.normal(size=8).astype(np.float32))
            texts.append(f"{names[m % 3]} item {m} number {i}")
            y.append(m % 3)
    x, y = np.array(x), np.array(y)
    dm = DMeans(3, Perfect(), seed=0)
    dm.fit(x, texts)
    d = semantic.diagnose(dm, x, y)
    assert 0.9 <= d["purity"] <= 1.0
    assert d["noul_auc"] > 0.95
    assert d["moved_wrong"] == 0
    assert 0.0 < d["pair_same_rate"] < 1.0


def _result(accs: dict[str, tuple[float, float]]) -> dict:
    """A fake semantic.json: per dataset, (best baseline acc, d-means acc)."""
    out = {"seeds": [0], "usd_per_decision": {"a": 1e-5}, "jev_answers": 1234, "datasets": {}}
    for name, (base, dm) in accs.items():
        row = lambda a: {"acc": a, "nmi": a, "ari": a}  # noqa: E731
        runs = {"k-means": [row(base)], "e-means": [row(base - 0.01)], "o-means": [row(base - 0.02)], "d-means": [{**row(dm), "decisions": 100.0, "purity": 0.9}]}
        out["datasets"][name] = {"n": 10, "k": 2, "runs": runs, "mean": {m: dict(r[0]) for m, r in runs.items()}}
    return out


def test_verify_flags_drift_but_not_losses():
    ref = _result({"a": (0.5, 0.6), "b": (0.7, 0.6)})
    assert semantic.verify(copy.deepcopy(ref), ref) == []
    assert semantic.losses(ref) == {"a": [], "b": ["acc", "nmi", "ari"]}
    drift = copy.deepcopy(ref)
    drift["datasets"]["a"]["runs"]["d-means"][0]["purity"] = 0.8
    assert any("purity" in e for e in semantic.verify(drift, ref))
    extra = _result({"a": (0.5, 0.6), "c": (0.5, 0.6)})
    assert semantic.verify(extra, ref) == ["c: not in the committed results"]


def test_summary_names_every_loss():
    res = _result({"a": (0.5, 0.6), "b": (0.7, 0.65)})
    text = report.summary(res)
    assert "on 1 of 2 datasets" in text and "It trails on b (ACC -5.0, NMI -5.0, ARI -5.0)" in text
    assert "1,234 decisions" in text
    table = report.semantic_table(res)
    assert "| a | 10 | 2 | 50.0 | 49.0 | 48.0 | **60.0** | +10.0 | $0.001 |" in table
    assert "| b | 10 | 2 | **70.0** |" in table


def test_readme_explains_each_dataset_d_means_loses_on():
    res = json.loads((semantic.OUT).read_text())
    readme = (ROOT / "README.md").read_text()
    if not any(semantic.losses(res).values()):
        return
    section = readme.split("### Where d-means loses", 1)[1].split("\n## ", 1)[0]
    prose = re.sub(r"<!-- bench:diagnose -->.*?<!-- /bench:diagnose -->", "", section, flags=re.S)
    for name, lost in semantic.losses(res).items():
        if lost:
            assert f"**{report.NAMES[name]}**" in prose, f"README gives no cause for {name}"


@pytest.mark.parametrize("name", semantic.DATASETS)
def test_committed_results_cover_every_dataset(name):
    res = json.loads((semantic.OUT).read_text())
    d = res["datasets"][name]
    assert len(d["runs"]["d-means"]) == len(semantic.SEEDS)
    assert set(semantic.BASELINES) <= set(d["runs"])
    assert {"purity", "decisions"} <= set(d["runs"]["d-means"][0])
    assert name in res["usd_per_decision"]
