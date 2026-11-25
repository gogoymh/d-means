"""Cluster cards: what the decision model reads about a cluster.

A card is the cluster's most distinctive words (class-based TF-IDF) and the texts closest to
its centroid. Cards are built from the data alone; the true labels never enter them.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .geometry import sqdist

EXAMPLE_CHARS = 160


class Vocabulary:
    def __init__(self, texts: list[str]):
        self.vec = CountVectorizer(stop_words="english", min_df=2, max_features=30000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        try:
            self.tf = self.vec.fit_transform(texts).tocsr()
        except ValueError:  # vocabulary too small (tiny test inputs)
            self.vec = CountVectorizer(min_df=1, token_pattern=r"(?u)\b\w+\b")
            self.tf = self.vec.fit_transform(texts).tocsr()
        self.words = np.array(self.vec.get_feature_names_out())

    def keywords(self, labels: np.ndarray, groups: int, top: int = 6) -> list[list[str]]:
        """Top words per group by class-based TF-IDF (term share in the group x rarity across groups)."""
        onehot = np.zeros((groups, len(labels)), dtype=np.float32)
        onehot[labels, np.arange(len(labels))] = 1.0
        tf = np.asarray((self.tf.T @ onehot.T).T)  # groups x vocab
        per_group = tf.sum(1, keepdims=True) + 1e-9
        freq = tf.sum(0) + 1e-9
        avg = per_group.mean()
        score = (tf / per_group) * np.log1p(avg / freq)[None, :]
        out = []
        for g in range(groups):
            idx = np.argsort(-score[g])[:top]
            out.append([str(self.words[i]) for i in idx if tf[g, i] > 0])
        return out


def _short(t: str) -> str:
    return t if len(t) <= EXAMPLE_CHARS else t[: EXAMPLE_CHARS - 3].rsplit(" ", 1)[0] + "..."


def build(vocab: Vocabulary, texts: list[str], x: np.ndarray, labels: np.ndarray, groups: int, examples: int = 4) -> list[dict]:
    """One card per group: {"keywords": "...", "examples": [...]} (empty groups get empty cards)."""
    kws = vocab.keywords(labels, groups)
    cards = []
    for g in range(groups):
        idx = np.flatnonzero(labels == g)
        if idx.size == 0:
            cards.append({"keywords": "", "examples": []})
            continue
        center = x[idx].mean(0, keepdims=True)
        near = idx[np.argsort(sqdist(x[idx], center)[:, 0])[:examples]]
        cards.append({"keywords": ", ".join(kws[g]), "examples": [_short(texts[i]) for i in near]})
    return cards


def option_names(cards: list[dict], ids: list[int]) -> list[str]:
    """Readable, unique Choice option names (Jev weighs the option name heavily)."""
    names, seen = [], set()
    for g in ids:
        words = cards[g]["keywords"].split(", ")[:3]
        base = " / ".join(w for w in words if w) or f"group {g}"
        name = base if base not in seen else f"{base} ({g})"
        seen.add(name)
        names.append(name)
    return names
