"""Client for Jev, TypeSafe's System One decision model, through OpenRouter's Decisions API.

Every request is cached by the SHA-256 of its canonical JSON, so a benchmark can be replayed
offline with exactly the answers Jev gave. In offline mode a cache miss raises CacheMiss.
"""

import gzip
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import _log

log = _log.get("jev")
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"


class CacheMiss(KeyError):
    pass


def request_key(model: str, req: dict) -> str:
    blob = json.dumps({"model": model, **req}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _compact(answers: dict) -> dict:
    out = {}
    for qid, a in answers.items():
        if a.get("type") == "noul":
            out[qid] = round(float(a["noul"]), 4)
        else:
            out[qid] = {o: round(float(p), 4) for o, p in a["probabilities"].items()}
    return out


class Jev:
    """Ask typed questions; answers come back compact: a float for a Noul, {option: prob} for a Choice."""

    def __init__(self, cache: str | Path | None = None, offline: bool = False, model: str = MODEL, workers: int = 16):
        self.model, self.offline, self.workers = model, offline, workers
        self.path = Path(cache) if cache else None
        self.store: dict[str, dict] = {}
        self.new: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.used: set[str] = set()
        self.stats = {"requests": 0, "cache_hits": 0, "api_calls": 0, "input_tokens": 0, "cost_usd": 0.0, "retries": 0}
        self.models_seen: set[str] = set()
        if self.path and self.path.exists():
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.store[row["k"]] = row["a"]
            log.info("loaded %d cached Jev answers from %s", len(self.store), self.path)

    def _call(self, req: dict) -> dict:
        body = json.dumps({"model": self.model, **req}).encode()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        delay = 1.0
        for attempt in range(8):
            r = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=90) as resp:
                    out = json.loads(resp.read())
                with self.lock:
                    self.stats["api_calls"] += 1
                    u = out.get("usage", {})
                    self.stats["input_tokens"] += int(u.get("input_tokens", 0))
                    self.stats["cost_usd"] += float(u.get("cost", 0.0))
                    self.models_seen.add(out.get("model", "?"))
                return _compact(out["answers"])
            except urllib.error.HTTPError as e:
                msg = e.read()[:300].decode("utf-8", "replace")
                if e.code == 402:
                    raise RuntimeError("Jev HTTP 402: the OpenRouter account has no credits left") from e
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"Jev HTTP {e.code}: {msg}") from e
                log.warning("Jev HTTP %d (attempt %d), retrying in %.0fs: %s", e.code, attempt + 1, delay, msg)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                log.warning("Jev network error (attempt %d), retrying in %.0fs: %s", attempt + 1, delay, e)
            with self.lock:
                self.stats["retries"] += 1
            time.sleep(delay)
            delay = min(delay * 2, 30)
        raise RuntimeError("Jev request failed after 8 attempts")

    def _one(self, req: dict) -> dict:
        k = request_key(self.model, req)
        with self.lock:
            self.used.add(k)
            self.stats["requests"] += 1
            hit = self.store.get(k)
            if hit is not None:
                self.stats["cache_hits"] += 1
                return hit
        if self.offline:
            raise CacheMiss(k)
        ans = self._call(req)
        with self.lock:
            self.store[k] = ans
            self.new[k] = ans
            flush = len(self.new) % 200 == 0
        if flush:  # keep paid answers even if a later request fails
            self.save()
        return ans

    def ask(self, reqs: list[dict]) -> list[dict]:
        """Answer requests of the form {"state": ..., "questions": {...}}, in order, concurrently."""
        if not reqs:
            return []
        t0 = time.time()
        keys = [request_key(self.model, r) for r in reqs]
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)  # identical requests in one batch are sent once
        uniq = list(first.values())
        with ThreadPoolExecutor(self.workers) as pool:
            got = dict(zip((keys[i] for i in uniq), pool.map(self._one, (reqs[i] for i in uniq))))
        with self.lock:
            self.stats["requests"] += len(reqs) - len(uniq)
            self.stats["cache_hits"] += len(reqs) - len(uniq)
        out = [got[k] for k in keys]
        log.info("Jev answered %d requests in %.1fs (cache hits so far %d, api calls %d)", len(reqs), time.time() - t0, self.stats["cache_hits"], self.stats["api_calls"])
        return out

    def save(self, keep: set[str] | None = None) -> None:
        """Write the cache. With `keep`, only those keys are written (drops answers no longer used)."""
        if not self.path:
            return
        with self.lock:
            rows = dict(self.store) if keep is None else {k: v for k, v in self.store.items() if k in keep}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{threading.get_ident()}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
            for k in sorted(rows):
                f.write(json.dumps({"k": k, "a": rows[k]}, separators=(",", ":")) + "\n")
        tmp.replace(self.path)
        log.info("saved %d Jev answers to %s", len(rows), self.path)
