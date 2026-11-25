"""d-means mechanics, offline, with scripted deciders standing in for Jev."""

import numpy as np
import pytest

from dmeans import DMeans, EMeans
from dmeans.geometry import nearest
from dmeans.metrics import accuracy

CLASSES = ["finance", "sport", "science", "music"]
SUBTOPICS = [["bank", "stock", "loan"], ["goal", "match", "coach"], ["atom", "cell", "orbit"], ["guitar", "album", "choir"]]


@pytest.fixture(scope="module")
def corpus():
    """4 classes x 3 sub-topics; sub-topics of different classes are interleaved in space,
    so geometry alone cannot tell the classes apart."""
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(12, 16)).astype(np.float32) * 3
    texts, labels, rows = [], [], []
    for m in range(12):
        cls, sub = m % 4, m // 4
        for i in range(60):
            texts.append(f"{CLASSES[cls]} {SUBTOPICS[cls][sub]} report number {i} {SUBTOPICS[cls][sub]} {CLASSES[cls]} news")
            labels.append(cls)
            rows.append(modes[m] + rng.normal(size=16).astype(np.float32))
    return np.array(rows), texts, np.array(labels)


class Oracle:
    """Answers from the true labels: what a perfect decision model would say."""

    def __init__(self, texts, labels):
        self.label = dict(zip(texts, labels))
        self.requests = []

    def _card_class(self, card):
        votes = [self.label[t] for t in card["examples"] if t in self.label]
        return np.bincount(votes, minlength=4).argmax()

    def ask(self, reqs):
        self.requests += reqs
        out = []
        for r in reqs:
            q = next(iter(r["questions"].values()))
            if q["type"] == "noul":
                same = self._card_class(r["state"]["group_a"]) == self._card_class(r["state"]["group_b"])
                out.append({"same": 0.95 if same else 0.05})
            else:
                truth = self.label[r["state"]["text"]]
                out.append({"group": {name: (1.0 if self._card_class(card) == truth else 0.0) for name, card in q["criteria"].items()}})
        return out


def test_decisions_fix_what_geometry_cannot(corpus):
    x, texts, y = corpus
    c, _ = EMeans(4, nepochs=5, seed=0).fit(x)
    geo = accuracy(y, nearest(x, c)[0])
    dm = DMeans(4, Oracle(texts, y), over=3, seed=0)
    lab = dm.fit(x, texts)
    assert geo < 0.8
    assert accuracy(y, lab) > 0.97
    assert accuracy(y, lab) > geo + 0.2


def test_requests_have_the_documented_shape(corpus):
    x, texts, y = corpus
    oracle = Oracle(texts, y)
    dm = DMeans(4, oracle, about="subject", budget=0.1, seed=1)
    dm.fit(x, texts)
    nouls = [r for r in oracle.requests if "same" in r["questions"]]
    choices = [r for r in oracle.requests if "group" in r["questions"]]
    assert len(nouls) == dm.report["pair_decisions"] > 0
    assert len(choices) == dm.report["point_decisions"] == round(0.1 * len(x))
    q = nouls[0]["questions"]["same"]
    assert q["type"] == "noul" and "subject" in q["instructions"] and set(q["criteria"]) == {"true", "false"}
    assert set(nouls[0]["state"]) == {"group_a", "group_b"}
    assert {"keywords", "examples"} <= set(nouls[0]["state"]["group_a"])
    c = choices[0]["questions"]["group"]
    assert c["type"] == "choice" and 2 <= len(c["criteria"]) <= 4
    assert choices[0]["state"]["text"] in texts


def test_zero_budget_asks_no_point_questions(corpus):
    x, texts, y = corpus
    oracle = Oracle(texts, y)
    dm = DMeans(4, oracle, budget=0.0, seed=2)
    lab = dm.fit(x, texts)
    assert dm.report["point_decisions"] == 0
    assert set(np.unique(lab)) <= set(range(4))
    assert np.array_equal(lab, dm.merge_labels)
