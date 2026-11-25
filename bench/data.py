"""Labeled text datasets for the benchmark, downloaded on demand and pinned by SHA-256.

Nothing here is redistributed: files land in .cache/ (git-ignored). Labels are used only to
score a finished clustering, never by an algorithm.
"""

import csv
import hashlib
import io
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dmeans import _log

log = _log.get("bench.data")
ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("DMEANS_CACHE", ROOT / ".cache"))
MAX_CHARS = 1000  # texts are cut here for both the embedder and the decision model


@dataclass
class Dataset:
    name: str
    texts: list[str]
    labels: np.ndarray
    about: str  # what the grouping is about, in a few words (d-means input; no label names)

    @property
    def k(self) -> int:
        return int(len(np.unique(self.labels)))


SOURCES = {
    "trec": dict(
        url="https://huggingface.co/datasets/SetFit/TREC-QC/resolve/main/test.jsonl",
        about="type of answer the question asks for",
    ),
    "stackexchange": dict(
        url="https://huggingface.co/datasets/mteb/stackexchange-clustering/resolve/main/valid.jsonl",
        about="subject area",
    ),
    "bbc-news": dict(
        url="https://huggingface.co/datasets/SetFit/bbc-news/resolve/main/test.jsonl",
        about="news topic",
    ),
    "ag-news": dict(
        url="https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        about="news topic",
    ),
    "20-newsgroups": dict(
        url="https://huggingface.co/datasets/SetFit/20_newsgroups/resolve/main/test.jsonl",
        about="discussion topic",
    ),
    "banking77": dict(
        url="https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        about="customer request",
    ),
}
SHA_FILE = Path(__file__).resolve().parent / "data.sha256.json"


def _pins() -> dict:
    return json.loads(SHA_FILE.read_text()) if SHA_FILE.exists() else {}


def _fetch(name: str) -> bytes:
    src = SOURCES[name]
    path = CACHE / "data" / f"{name}{Path(src['url']).suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s from %s", name, src["url"])
        req = urllib.request.Request(src["url"], headers={"User-Agent": "d-means-bench/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    pin = _pins().get(name)
    if pin and pin != digest:
        raise RuntimeError(f"{name}: sha256 {digest} does not match pinned {pin}; delete {path} and retry")
    return data


def _clean(t: str) -> str:
    return " ".join(t.split())[:MAX_CHARS]


def load(name: str) -> Dataset:
    raw = _fetch(name)
    texts: list[str] = []
    labels: list[str] = []
    if name in ("trec", "bbc-news", "20-newsgroups"):
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line)
            lab = row["label_coarse_text"] if name == "trec" else row["label_text"]
            texts.append(row["text"])
            labels.append(str(lab))
    elif name == "stackexchange":  # first split of the MTEB validation set, 3,000 titles sampled
        first = json.loads(raw[: raw.index(b"\n")].decode("utf-8"))
        pick = np.sort(np.random.default_rng(0).choice(len(first["sentences"]), 3000, replace=False))
        texts = [first["sentences"][i] for i in pick]
        labels = [first["labels"][i].split(".")[0] for i in pick]
    elif name == "ag-news":
        for cls, title, desc in csv.reader(io.StringIO(raw.decode("utf-8"))):
            texts.append(f"{title}. {desc}".replace("\\", " "))
            labels.append(cls)
    elif name == "banking77":
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        texts = [r["text"] for r in rows]
        labels = [r["category"] for r in rows]
    else:
        raise KeyError(name)
    texts = [_clean(t) for t in texts]
    keep = [i for i, t in enumerate(texts) if len(t) >= 10]  # 20NG has empty posts
    texts = [texts[i] for i in keep]
    _, y = np.unique(np.array([labels[i] for i in keep]), return_inverse=True)
    ds = Dataset(name, texts, y.astype(np.int64), SOURCES[name]["about"])
    log.info("loaded %s: n=%d k=%d (dropped %d empty)", name, len(texts), ds.k, len(labels) - len(keep))
    return ds


def pin_all() -> dict:
    """Write the SHA-256 of every downloaded file to data.sha256.json."""
    out = {}
    for name, src in SOURCES.items():
        _fetch(name)
        path = CACHE / "data" / f"{name}{Path(src['url']).suffix}"
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    SHA_FILE.write_text(json.dumps(out, indent=1) + "\n")
    return out
