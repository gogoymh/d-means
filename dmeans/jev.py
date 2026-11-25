"""Client for Jev, TypeSafe's System One decision model, through OpenRouter's Decisions API.

Every request is cached by the SHA-256 of its canonical JSON, so a benchmark can be replayed
offline with exactly the answers Jev gave. In offline mode a cache miss raises CacheMiss.
Other decision models live in dmeans.deciders and share the same cache and answer shape.
"""

import time  # noqa: F401  (tests patch jev.time.sleep to skip backoff)
from pathlib import Path

from . import _log
from .deciders import CachedDecider, CacheMiss, request_key

__all__ = ["Jev", "CacheMiss", "request_key", "URL", "MODEL"]
log = _log.get("jev")
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"


def _compact(answers: dict) -> dict:
    out = {}
    for qid, a in answers.items():
        if a.get("type") == "noul":
            out[qid] = round(float(a["noul"]), 4)
        else:
            out[qid] = {o: round(float(p), 4) for o, p in a["probabilities"].items()}
    return out


class Jev(CachedDecider):
    """Ask typed questions; answers come back compact: a float for a Noul, {option: prob} for a Choice."""

    label = "Jev"

    def __init__(self, cache: str | Path | None = None, offline: bool = False, model: str = MODEL, workers: int = 16):
        super().__init__(model, cache, offline, workers)

    def _call(self, req: dict) -> dict:
        return _compact(self._post(URL, {"model": self.model, **req}, "OPENROUTER_API_KEY")["answers"])
