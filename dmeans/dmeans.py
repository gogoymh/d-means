"""d-means: k-means steered by a typed decision model.

Geometry is good at "how far", bad at "what kind". k-means, e-means and o-means all cut the
embedding space by distance alone, so they merge two categories that sit close together and
split one that is spread out, and they guess on points that sit between two centroids.
d-means keeps the fast geometry and hands the two decisions geometry cannot make to a
decision model (any `ask(requests) -> answers`, see dmeans.deciders), which reads the texts:

1. Over-cluster. e-means splits the data into `over * k` micro-clusters, which are much purer
   than k clusters.
2. Merge by decision. For each micro-cluster and its nearest neighbours, the decider answers a Noul:
   do these two groups share one topic? Pairs it was not asked about get a prior fitted on
   the pairs it was asked about (isotonic in centroid cosine). Average-linkage on 1 - p joins
   the micro-clusters into exactly k groups.
3. Settle the borderline points. The `budget` share of points with the smallest margin between
   their two nearest groups get a Choice question over their nearest `candidates` groups. The decider's
   probabilities are multiplied with the geometric posterior and the point takes the argmax.

Every call goes through a decider with `ask(requests) -> answers` (see dmeans.deciders): any
chat LLM, Jev, Clef, or a plain function `fn(state, question)`. Answers are checked and coerced into one shape,
so the algorithm is tested offline with a scripted decider and replayed offline from cached answers.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.isotonic import IsotonicRegression

from . import _log, cards
from .deciders import as_decider, normalize
from .emeans import EMeans
from .geometry import nearest, sqdist

log = _log.get("dmeans")



def merge_question(about: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Do the texts in `group_a` and the texts in `group_b` share the same {about}, so that one category label fits both groups?",
        "criteria": {
            "true": f"Both groups share one {about}; a person sorting the texts by {about} would file them together.",
            "false": f"The two groups differ in {about}.",
        },
    }


def assign_question(about: str, criteria: dict) -> dict:
    return {
        "type": "choice",
        "instructions": f"Which group shares the {about} of `text`? Each option lists a group's typical words and example texts.",
        "criteria": criteria,
    }


class DMeans:
    name = "d-means"

    def __init__(
        self,
        k: int,
        decider,
        about: str = "topic",
        over: int = 3,
        neighbors: int = 6,
        budget: float = 0.15,
        candidates: int = 5,
        nepochs: float = 5,
        seed: int = 1234,
        linkage: str = "average",
        fusion: str = "product",
    ):
        self.k, self.decider, self.about, self.over, self.neighbors = k, decider, about, over, neighbors
        self.budget, self.candidates, self.nepochs, self.seed = budget, candidates, nepochs, seed
        self.linkage, self.fusion = linkage, fusion
        self.decider = as_decider(decider)
        log.info("d-means decider: %s %s", type(self.decider).__name__, getattr(self.decider, "model", ""))

    def _ask(self, reqs: list[dict]) -> list[dict]:
        return normalize(reqs, self.decider.ask(reqs)) if reqs else []

    # --- stage 1 ---------------------------------------------------------------------------
    def _micro(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k0 = int(min(self.over * self.k, max(self.k, len(x) // 4)))
        self.engine = EMeans(k0, nepochs=self.nepochs, seed=self.seed)
        mc, _ = self.engine.fit(x)
        ml, _ = nearest(x, mc)
        used = np.unique(ml)  # drop micro-clusters that ended empty
        remap = np.full(k0, -1)
        remap[used] = np.arange(used.size)
        return mc[used], remap[ml]

    # --- stage 2 ---------------------------------------------------------------------------
    def _pairs(self, mc: np.ndarray) -> list[tuple[int, int]]:
        d = sqdist(mc, mc)
        np.fill_diagonal(d, np.inf)
        p = min(self.neighbors, len(mc) - 1)
        nbr = np.argsort(d, axis=1)[:, :p]
        return sorted({(min(i, j), max(i, j)) for i in range(len(mc)) for j in nbr[i]})

    def _merge(self, mc: np.ndarray, micro_cards: list[dict]) -> np.ndarray:
        m = len(mc)
        if m <= self.k:
            return np.arange(m)
        pairs = self._pairs(mc)
        reqs = [
            {
                "state": {"group_a": micro_cards[i], "group_b": micro_cards[j]},
                "questions": {"same": merge_question(self.about)},
            }
            for i, j in pairs
        ]
        ans = self._ask(reqs)
        p_same = np.array([float(a["same"]) for a in ans])
        unit = mc / np.maximum(np.linalg.norm(mc, axis=1, keepdims=True), 1e-12)
        cos = unit @ unit.T
        ii, jj = np.array(pairs).T
        prior = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        prior.fit(cos[ii, jj], p_same)
        aff = prior.predict(cos.ravel()).reshape(m, m)
        aff[ii, jj] = p_same
        aff[jj, ii] = p_same
        dist = 1.0 - aff
        np.fill_diagonal(dist, 0.0)
        if self.linkage == "spectral":
            from sklearn.cluster import SpectralClustering

            group = SpectralClustering(self.k, affinity="precomputed", random_state=self.seed, assign_labels="cluster_qr").fit_predict(np.clip(aff, 1e-6, 1.0))
        else:
            group = AgglomerativeClustering(n_clusters=self.k, metric="precomputed", linkage=self.linkage).fit_predict(dist)
        self.pairs, self.p_same = pairs, p_same
        log.info("merged %d micro-clusters into %d groups with %d pair decisions", m, self.k, len(pairs))
        return group

    # --- stage 3 ---------------------------------------------------------------------------
    def _group_dist(self, x: np.ndarray, mc: np.ndarray, group: np.ndarray) -> np.ndarray:
        d = sqdist(x, mc)
        out = np.full((len(x), self.k), np.inf, dtype=np.float32)
        for g in range(self.k):
            members = np.flatnonzero(group == g)
            if members.size:
                out[:, g] = d[:, members].min(1)
        return out

    def _settle(self, x: np.ndarray, texts: list[str], gd: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n = len(x)
        q = int(round(self.budget * n))
        if q == 0 or self.k < 2:
            return labels
        srt = np.sort(gd, axis=1)
        margin = srt[:, 1] - srt[:, 0]
        tau = float(np.median(margin)) or 1e-6
        ask = np.argsort(margin, kind="stable")[:q]
        group_cards = cards.build(self.vocab, texts, x, labels, self.k)
        rng = np.random.default_rng(self.seed + 7)
        m = min(self.candidates, self.k)
        reqs, opts = [], []
        for i in ask:
            cand = [int(g) for g in np.argsort(gd[i])[:m] if np.isfinite(gd[i, g])]
            cand = [cand[j] for j in rng.permutation(len(cand))]  # decision models are sensitive to option order
            names = cards.option_names(group_cards, cand)
            opts.append((cand, names))
            reqs.append(
                {
                    "state": {"text": texts[i]},
                    "questions": {"group": assign_question(self.about, {nm: group_cards[g] for nm, g in zip(names, cand)})},
                }
            )
        ans = self._ask(reqs)
        out = labels.copy()
        changed = 0
        for i, (cand, names), a in zip(ask, opts, ans):
            p_dec = np.array([a["group"].get(nm, 0.0) for nm in names]) + 1e-6
            g_d = gd[i, cand]
            p_geo = np.exp(-(g_d - g_d.min()) / tau)
            post = p_dec * p_geo if self.fusion == "product" else p_dec
            new = cand[int(np.argmax(post))]
            changed += int(new != out[i])
            out[i] = new
        self.asked, self.changed = ask, changed
        log.info("settled %d borderline points with Choice decisions, %d changed group", q, changed)
        return out

    # --- fit -------------------------------------------------------------------------------
    def fit(self, x: np.ndarray, texts: list[str]) -> np.ndarray:
        """Cluster rows of `x` (embeddings of `texts`) into k groups; returns one label per row."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if len(texts) != len(x):
            raise ValueError("x and texts must have the same length")
        self.pairs, self.p_same, self.asked, self.changed = [], np.zeros(0), np.zeros(0, dtype=np.int64), 0
        self.vocab = cards.Vocabulary(texts)
        mc, ml = self._micro(x)
        micro_cards = cards.build(self.vocab, texts, x, ml, len(mc))
        group = self._merge(mc, micro_cards)
        gd = self._group_dist(x, mc, group)
        labels = gd.argmin(1)
        self.merge_labels = labels.copy()
        labels = self._settle(x, texts, gd, labels)
        self.micro_centroids, self.micro_group, self.labels = mc, group, labels
        self.report = {
            "micro_clusters": int(len(mc)),
            "pair_decisions": len(self.pairs),
            "point_decisions": int(len(self.asked)),
            "decisions": len(self.pairs) + int(len(self.asked)),
            "decisions_per_point": (len(self.pairs) + len(self.asked)) / len(x),
            "points_moved": int(self.changed),
            "engine_epochs": self.engine.epochs_run,
        }
        log.info("d-means done: %s", self.report)
        return labels
