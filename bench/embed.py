"""Sentence embeddings through OpenRouter, cached in .cache/emb as float16."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dmeans import _log

from .data import CACHE

log = _log.get("bench.embed")
MODEL = "sentence-transformers/all-mpnet-base-v2"
URL = "https://openrouter.ai/api/v1/embeddings"


def _post(batch: list[str], model: str) -> list[list[float]]:
    body = json.dumps({"model": model, "input": batch}).encode()
    for attempt in range(6):
        req = urllib.request.Request(
            URL,
            data=body,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            vecs = [row["embedding"] for row in sorted(out["data"], key=lambda r: r["index"])]
            if len(vecs) == len(batch):
                return vecs
            log.warning("embedding batch returned %d of %d rows, retrying", len(vecs), len(batch))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            log.warning("embedding request failed (attempt %d): %s", attempt + 1, e)
        time.sleep(2**attempt)
    raise RuntimeError("embedding request failed after 6 attempts")


def embed(name: str, texts: list[str], model: str = MODEL, batch: int = 128) -> np.ndarray:
    """L2-normalized float32 embeddings, one row per text."""
    digest = hashlib.sha256(("\n".join(texts) + model).encode()).hexdigest()[:16]
    path = CACHE / "emb" / f"{name}-{digest}.npy"
    if path.exists():
        x = np.load(path).astype(np.float32)
    else:
        log.info("embedding %s: %d texts with %s", name, len(texts), model)
        chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]
        with ThreadPoolExecutor(4) as pool:
            parts = list(pool.map(lambda c: _post(c, model), chunks))
        x = np.asarray([v for p in parts for v in p], dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, x.astype(np.float16))
        x = x.astype(np.float16).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x
