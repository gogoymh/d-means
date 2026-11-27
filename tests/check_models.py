"""Exit 0 only if every decision model in the README study has a committed run of the default strategy.

    uv run python tests/check_models.py               # every model on all six datasets
    uv run python tests/check_models.py trec          # every model on at least these

Each run must use the K-means start (10 restarts) on seeds 0, 1, 2, come with its own committed answer
cache so the README table replays offline, and beat the best baseline on ACC on every dataset it covers.
"""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.report import STUDY, current_results  # noqa: E402
from bench.semantic import BASELINES, DATASETS, MODELS, paths  # noqa: E402


def main(argv: list[str]) -> int:
    required = argv[0].split(",") if argv else DATASETS
    got = current_results()
    errors = []
    for model in STUDY:
        res = got.get(model)
        if res is None:
            errors.append(f"{model}: no results file with strategy 'kmeans' at {paths(model, current=True)[1].name}")
            continue
        if res.get("kmeans_restarts") != 10 or res.get("seeds") != [0, 1, 2]:
            errors.append(f"{model}: restarts {res.get('kmeans_restarts')}, seeds {res.get('seeds')}")
        missing = [d for d in required if d not in res["datasets"]]
        if missing:
            errors.append(f"{model}: missing {missing}")
        answers = paths(model, current=True)[0]
        n = 0
        if answers.exists():
            with gzip.open(answers, "rt") as f:
                n = sum(1 for _ in f)
        if n < res.get("jev_answers", 1):
            errors.append(f"{model}: {n} cached answers in {answers.name}, run used {res.get('jev_answers')}")
        for d, r in res["datasets"].items():
            best = max(r["mean"][b]["acc"] for b in BASELINES)
            if r["mean"]["d-means"]["acc"] <= best:
                errors.append(f"{model} {d}: d-means {r['mean']['d-means']['acc']:.4f} <= best baseline {best:.4f}")
        print(f"{MODELS[model]:18s} {len(res['datasets'])} datasets  {n:,} answers")
    for e in errors:
        print("FAIL", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
