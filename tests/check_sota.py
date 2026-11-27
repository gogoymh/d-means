"""Exit 0 only if the committed Clef run of the default strategy beats the best baseline on ACC everywhere.

    uv run python tests/check_sota.py
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results" / "semantic-clef.json"
BASELINES = ["k-means", "e-means", "o-means"]
DATASETS = ["trec", "stackexchange", "ag-news", "bbc-news", "20-newsgroups", "banking77"]


def main() -> int:
    res = json.loads(RESULTS.read_text())
    errors = []
    if res.get("strategy") != "kmeans" or res.get("kmeans_restarts") != 10:
        errors.append(f"not the default strategy with a 10-restart K-means baseline: {res.get('strategy')}, {res.get('kmeans_restarts')}")
    for name in DATASETS:
        d = res["datasets"].get(name)
        if d is None:
            errors.append(f"{name}: missing")
            continue
        mean = d["mean"]
        best = max(mean[b]["acc"] for b in BASELINES)
        dm = mean["d-means"]["acc"]
        print(f"{name:14s} best baseline {100 * best:5.1f}  d-means {100 * dm:5.1f}  {100 * (dm - best):+.1f}")
        if dm <= best:
            errors.append(f"{name}: d-means {dm:.4f} <= best baseline {best:.4f}")
    for e in errors:
        print("FAIL", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
