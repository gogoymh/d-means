"""Convergence benchmark: clustering accuracy against epochs for k-means, e-means, o-means and d-means.

    uv run python -m bench.convergence --write    # run, save results/convergence.json
    uv run python -m bench.report                 # redraw the chart and the README block
    uv run python -m bench.convergence --verify   # rerun offline, fail on drift (or if d-means is not fastest on topics)

An epoch is one pass over the data by the geometric engine: one Lloyd iteration for k-means, the
epoch budget of e-means and o-means, and the epoch budget of the e-means run that builds d-means'
micro-clusters. Each method is run once per budget, as the e-means paper does, so every point is
a finished run. d-means also asks decisions; their count per point is recorded next to its curve.

The default problem, `topics`, is synthetic and needs no network: 6 topics, each spread over 4
sub-topic modes placed at random, so modes of different topics sit next to each other and
distance alone cannot tell the topics apart. Its decisions come from `SimulatedDecider`, which
reads the hidden topic behind each text and gives a wrong answer 20% of the time. It stands in
for Jev so the chart can be rebuilt offline. `--problems trec,stackexchange --seeds 0,1,2 --live --write`
draws the same curves on real texts with Jev (needs OpenRouter credits, about $1).
"""

import argparse
import json
import re
import sys
import time

import numpy as np

from dmeans import DMeans, _log, lloyd
from dmeans.geometry import nearest
from dmeans.deciders import CacheMiss, Decisions, request_key
from dmeans.metrics import accuracy

from . import official
from .semantic import MODEL, RESULTS

log = _log.get("bench.convergence")
OUT = RESULTS / "convergence.json"
ANSWERS = RESULTS / "jev-convergence.jsonl.gz"  # kept apart: bench.semantic --write prunes its own cache
METHODS = ["k-means", "e-means", "o-means", "d-means"]
BASELINES = METHODS[:-1]
EPOCHS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
SEEDS = list(range(50))
ERROR = 0.2


# --- the offline problem ----------------------------------------------------------------------
def _word(i: int) -> str:
    """A letters-only token for integer i ("ba", "bb", ...), so card keywords stay readable."""
    s = ""
    i += 26
    while i:
        i, r = divmod(i, 26)
        s = "abcdefghijklmnopqrstuvwxyz"[r] + s
    return s


def topics(n: int = 3000, d: int = 32, k: int = 6, modes_per_topic: int = 4, seed: int = 0) -> dict:
    """Synthetic labeled texts: k topics, each a union of `modes_per_topic` Gaussian modes at random places.

    Rows are the same mixture as bench.mse (unit-normal mode centers, noise 0.6). Each text names
    its sub-topic with three words from that sub-topic's own vocabulary; nothing in a text names
    the topic, so cards and decisions see only what a reader would.
    """
    rng = np.random.default_rng(seed)
    modes = k * modes_per_topic
    centers = rng.normal(size=(modes, d)).astype(np.float32)
    mode = rng.integers(0, modes, n)
    x = centers[mode] + 0.6 * rng.normal(size=(n, d)).astype(np.float32)
    vocab = [[_word(m * 8 + j) for j in range(8)] for m in range(modes)]
    texts = [f"item {_word(i)} " + " ".join(rng.choice(vocab[m], 3, replace=False)) for i, m in enumerate(mode)]
    return {"x": x, "texts": texts, "labels": (mode % k).astype(np.int64), "k": k, "about": "topic"}


class SimulatedDecider:
    """Answers d-means' questions from the hidden labels, wrong on `error` of its answers.

    A card's topic is the majority label of its example texts. A Noul says "same" when both cards
    share it; a Choice picks an option whose card has the text's topic (a random option when none
    has it). With probability `error` the answer is flipped: "same" becomes "different", the pick
    moves to another option. Probabilities are calibrated to that error rate. The coin is seeded
    by the request itself, so answers do not depend on call order and replays are exact.
    """

    def __init__(self, texts: list[str], labels: np.ndarray, error: float = ERROR, seed: int = 0):
        self.label = dict(zip(texts, labels.tolist()))
        self.error, self.seed = error, seed
        self.stats = {"requests": 0, "flipped": 0}

    def _topic(self, card: dict) -> int:
        votes = [self.label[t] for t in card["examples"] if t in self.label]
        return int(np.bincount(votes).argmax()) if votes else -1

    def _one(self, req: dict) -> dict:
        rng = np.random.default_rng([self.seed, int(request_key("simulated", req), 16)])
        flip = bool(rng.random() < self.error)
        self.stats["requests"] += 1
        self.stats["flipped"] += flip
        qid, q = next(iter(req["questions"].items()))
        if q["type"] == "noul":
            same = self._topic(req["state"]["group_a"]) == self._topic(req["state"]["group_b"])
            return {qid: 1.0 - self.error if same != flip else self.error}
        names = list(q["criteria"])
        truth = self.label[req["state"]["text"]]
        right = [nm for nm in names if self._topic(q["criteria"][nm]) == truth] or [names[int(rng.integers(len(names)))]]
        pick = right[0]
        if flip and len(names) > 1:
            others = [nm for nm in names if nm != pick]
            pick = others[int(rng.integers(len(others)))]
        rest = self.error / max(len(names) - 1, 1)
        return {qid: {nm: (1.0 - self.error if nm == pick else rest) for nm in names}}

    def ask(self, reqs: list[dict]) -> list[dict]:
        return [self._one(r) for r in reqs]


# --- curves -----------------------------------------------------------------------------------
def _real(name: str, live: bool) -> tuple[dict, Decisions]:
    from . import data, embed

    ds = data.load(name)
    prob = {"x": embed.embed(name, ds.texts), "texts": ds.texts, "labels": ds.labels, "k": ds.k, "about": ds.about}
    return prob, Decisions(MODEL, ANSWERS, offline=not live)


def curves(prob: dict, decider, seed: int, epochs: list[float]) -> dict:
    """Accuracy of each method after each epoch budget, for one seed.

    k-means is Lloyd from k-means++ seeds; its seeding passes are not counted, which favours it.
    """
    x, y, k = prob["x"], prob["labels"], prob["k"]
    out = {"epochs": {m: [] for m in METHODS}, "passes": {m: [] for m in METHODS}, "acc": {m: [] for m in METHODS}, "decisions_per_point": []}

    def add(m: str, t: float, passes: float, acc: float) -> None:
        out["epochs"][m].append(t)
        out["passes"][m].append(passes)
        out["acc"][m].append(acc)

    whole = [t for t in epochs if t >= 1 and float(t).is_integer()]
    acc_at: dict[int, float] = {}
    lloyd(x, k, iters=int(max(whole)), seed=seed, init="k-means++", on_epoch=lambda it, c: acc_at.__setitem__(it, accuracy(y, nearest(x, c)[0])))
    for t in whole:
        add("k-means", t, float(t), acc_at[int(t)])
    for t in epochs:
        for name, train in (("e-means", official.emeans), ("o-means", official.omeans)):
            c, passes = train(x, k, nepochs=t, seed=seed)
            add(name, t, passes, accuracy(y, nearest(x, c)[0]))
        dm = DMeans(k, decider, about=prob["about"], nepochs=t, seed=seed, consensus=False, start="merge", budget=0.15, refine=False)  # recorded with the v1 strategy
        lab = dm.fit(x, prob["texts"])
        add("d-means", t, dm.report["engine_epochs"], accuracy(y, lab))
        out["decisions_per_point"].append(dm.report["decisions_per_point"])
    log.info("seed %d: final acc k-means %.3f, e-means %.3f, o-means %.3f, d-means %.3f", seed, *(out["acc"][m][-1] for m in METHODS))
    return out


def summarize(runs: list[dict]) -> dict:
    """Seeds stacked, mean accuracy per budget, the target and the epochs each method needs to reach it.

    The target is the best final accuracy of any baseline (its mean over seeds at the largest budget).
    A method reaches it at the first budget whose mean accuracy is at least the target, and the
    epochs it took are the passes over the data that run actually made.
    """
    acc = {m: [r["acc"][m] for r in runs] for m in METHODS}
    mean = {m: [float(v) for v in np.mean(acc[m], axis=0)] for m in METHODS}
    passes = runs[0]["passes"]
    best = max(BASELINES, key=lambda m: mean[m][-1])
    target = {"method": best, "acc": mean[best][-1], "epochs": runs[0]["epochs"][best][-1]}
    reach = {m: next((x for x, v in zip(passes[m], mean[m]) if v >= target["acc"] - 1e-12), None) for m in METHODS}
    return {
        "epochs": runs[0]["epochs"],
        "passes": passes,
        "mean": mean,
        "decisions_per_point": float(np.mean([r["decisions_per_point"] for r in runs])),
        "target": target,
        "epochs_to_target": reach,
        "acc": acc,
    }


def fastest_problems(result: dict) -> list[str]:
    """Problems where d-means reaches the target in fewer epochs than every baseline."""
    ok = []
    for name, p in result["problems"].items():
        reach = p["epochs_to_target"]
        d = reach["d-means"]
        if d is not None and all(reach[m] is None or d < reach[m] for m in BASELINES):
            ok.append(name)
    return ok


def run(names: list[str], seeds: list[int], epochs: list[float], live: bool = False) -> dict:
    out = {"problems": {}}
    for name in names:
        jev = None
        if name == "topics":
            prob = topics()
            decider = SimulatedDecider(prob["texts"], prob["labels"])
            meta = {"source": "synthetic", "decider": f"simulated, wrong on {ERROR:.0%} of answers"}
        else:
            prob, jev = _real(name, live)
            decider = jev
            meta = {"source": name, "decider": jev.model}
        try:
            runs = [curves(prob, decider, s, epochs) for s in seeds]
        finally:
            if jev is not None and live:
                jev.save()
        p = {**meta, "n": len(prob["texts"]), "k": prob["k"], "seeds": seeds}
        if isinstance(decider, SimulatedDecider):
            p["decider_flip_rate"] = decider.stats["flipped"] / max(decider.stats["requests"], 1)
        p.update(summarize(runs))
        out["problems"][name] = p
        log.info("%s: target %.3f (%s final); epochs to reach it %s", name, p["target"]["acc"], p["target"]["method"], p["epochs_to_target"])
    return out


def verify(result: dict, reference: dict) -> list[str]:
    """Problems found: accuracy drift from the committed results, or d-means not the fastest on the synthetic problem.

    Real datasets are reported as measured, so on them only drift counts.
    """
    errors = []
    for name, got in result["problems"].items():
        ref = reference["problems"].get(name)
        if ref is None:
            errors.append(f"{name}: not in the committed results")
            continue
        for m in METHODS:
            for i, (a, b) in enumerate(zip(ref["acc"][m], got["acc"][m])):
                for t, va, vb in zip(ref["epochs"][m], a, b):
                    if abs(va - vb) > 1e-9:
                        errors.append(f"{name}/{m}/seed{got['seeds'][i]}@{t:g}: acc {vb:.6f} != committed {va:.6f}")
        if got["epochs_to_target"] != ref["epochs_to_target"]:
            errors.append(f"{name}: epochs to target {got['epochs_to_target']} != committed {ref['epochs_to_target']}")
    slow = {n for n, p in result["problems"].items() if p["source"] == "synthetic"} - set(fastest_problems(result))
    errors += [f"{name}: d-means does not reach the target first ({result['problems'][name]['epochs_to_target']})" for name in sorted(slow)]
    return errors


def dumps(result: dict) -> str:
    """JSON with one line per list of numbers, so the file stays small and diffs stay readable."""
    text = json.dumps(result, indent=1)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="topics", help="topics (offline) and/or bench.data datasets")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--live", action="store_true", help="ask Jev for answers missing from the cache (real datasets)")
    ap.add_argument("--write", action="store_true", help="save results/convergence.json")
    ap.add_argument("--verify", action="store_true", help="fail unless results match results/convergence.json (and d-means is fastest on topics)")
    a = ap.parse_args(argv)
    _log.setup()
    t0 = time.perf_counter()
    try:
        result = run(a.problems.split(","), [int(s) for s in a.seeds.split(",")], EPOCHS, live=a.live)
    except CacheMiss as e:
        log.error("Jev answer %s is not in %s; run with --live to ask Jev", e, ANSWERS)
        return 2
    for name, p in result["problems"].items():
        print(f"\n{name}  (n={p['n']}, k={p['k']}, mean of {len(p['seeds'])} seeds; target {100 * p['target']['acc']:.1f}% = {p['target']['method']} final)")
        print(f"  {'method':10s} " + " ".join(f"{t:>7.3g}" for t in EPOCHS) + "   epochs to target")
        for m in METHODS:
            row = dict(zip(p["epochs"][m], p["mean"][m]))
            cells = " ".join(f"{100 * row[t]:7.1f}" if t in row else f"{'-':>7s}" for t in EPOCHS)
            reach = p["epochs_to_target"][m]
            print(f"  {m:10s} {cells}   {'never' if reach is None else f'{reach:.3g}'}")
    if a.write:  # merge by problem, so a run of one dataset keeps the others
        old = json.loads(OUT.read_text()) if OUT.exists() else {"problems": {}}
        OUT.write_text(dumps({"problems": {**old["problems"], **result["problems"]}}))
        log.info("wrote %s", OUT)
    if a.verify:
        errors = verify(result, json.loads(OUT.read_text()))
        for e in errors:
            log.error("verify: %s", e)
        if errors:
            return 1
        log.info("verify: results match %s for %s; d-means reaches the target first on %s (%.1fs)", OUT.name, ", ".join(result["problems"]), ", ".join(fastest_problems(result)) or "none", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
