"""Decision models for d-means.

d-means asks typed questions and needs answers back in one compact shape:

    request = {"state": {...}, "questions": {qid: {"type": "noul" | "choice", "instructions": ..., "criteria": {...}}}}
    answer  = {qid: float in [0, 1]}            for a Noul (probability the statement is true)
              {qid: {option: probability}}      for a Choice (options are the keys of `criteria`)

Anything with `ask(requests) -> answers` is a decider. d-means depends on none of them by name;
these come built in, side by side, and take the model as a string:

    from dmeans.deciders import Decisions, LLM

- `Decisions`: a typed decision model (TypeSafe's Jev, Cloudflare's Clef, ...) through OpenRouter's Decisions API.
- `LLM`: any chat model behind an OpenAI-compatible endpoint (OpenRouter, OpenAI, vLLM, Ollama, ...).
- A plain function `fn(state, question) -> answer`, wrapped by `as_decider`.

`CachedDecider` gives every remote model the same cache, offline replay and concurrency.
"""

import gzip
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import _log

log = _log.get("deciders")
RETRY_CODES = (408, 429, 500, 502, 503, 504)
__all__ = ["Decider", "CachedDecider", "FunctionDecider", "Decisions", "LLM", "CacheMiss", "as_decider", "coerce", "normalize", "request_key"]


class CacheMiss(KeyError):
    pass


@runtime_checkable
class Decider(Protocol):
    def ask(self, reqs: list[dict]) -> list[dict]: ...


def request_key(model: str, req: dict) -> str:
    blob = json.dumps({"model": model, **req}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# --- answers ---------------------------------------------------------------------------------
def _num(v) -> float | None:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes"):
            return 1.0
        if s in ("false", "no"):
            return 0.0
        try:
            return float(s[:-1]) / 100 if s.endswith("%") else float(s)
        except ValueError:
            return None
    if v is None or isinstance(v, (dict, list, tuple)):
        return None
    try:
        f = float(v)  # bool, int, float and numpy scalars
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def coerce(question: dict, value, qid: str = "?") -> float | dict:
    """Turn a loose answer (bool, "yes", 87, an option name, unnormalised scores) into the compact shape."""
    if question.get("type") == "noul":
        if isinstance(value, dict):
            value = next((value[k] for k in ("true", "noul", "probability", "p") if k in value), None)
        p = _num(value)
        if p is None:
            raise ValueError(f"answer to Noul '{qid}' is not a probability: {value!r}")
        if p > 1.0 and p <= 100.0:
            p /= 100.0  # a percentage
        return round(min(max(p, 0.0), 1.0), 4)
    options = list(question.get("criteria", {}))
    if isinstance(value, dict) and isinstance(value.get("probabilities"), dict):
        value = value["probabilities"]
    if isinstance(value, str):
        if value not in options:
            raise ValueError(f"answer to Choice '{qid}' names no option: {value!r}")
        value = {value: 1.0}
    if not isinstance(value, dict):
        raise ValueError(f"answer to Choice '{qid}' is not an option map: {value!r}")
    probs = {o: max(_num(value.get(o)) or 0.0, 0.0) for o in options}
    total = sum(probs.values())
    if total <= 0:
        return {o: round(1.0 / len(options), 4) for o in options} if options else {}
    return {o: round(p / total, 4) for o, p in probs.items()}


def normalize(reqs: list[dict], answers: list[dict]) -> list[dict]:
    """Check that every question got an answer and coerce each into the compact shape."""
    if not isinstance(answers, list) or len(answers) != len(reqs):
        got = len(answers) if isinstance(answers, list) else type(answers).__name__
        raise ValueError(f"decider returned {got} answers for {len(reqs)} requests")
    out = []
    for i, (r, a) in enumerate(zip(reqs, answers)):
        if not isinstance(a, dict):
            raise ValueError(f"decider answer {i} is {type(a).__name__}, expected a dict of question ids")
        row = {}
        for qid, q in r["questions"].items():
            if qid not in a:
                raise ValueError(f"decider answer {i} has no answer to question '{qid}'")
            row[qid] = coerce(q, a[qid], qid)
        out.append(row)
    return out


# --- adapters --------------------------------------------------------------------------------
class FunctionDecider:
    """Wrap `fn(state, question) -> answer`; called once per question."""

    def __init__(self, fn: Callable[[dict, dict], object]):
        self.fn = fn
        self.model = getattr(fn, "__name__", "function")

    def ask(self, reqs: list[dict]) -> list[dict]:
        return [{qid: coerce(q, self.fn(r["state"], q), qid) for qid, q in r["questions"].items()} for r in reqs]


def as_decider(d) -> Decider:
    """Accept an object with `ask`, or a function `fn(state, question) -> answer`."""
    if callable(getattr(d, "ask", None)):
        return d
    if callable(d):
        return FunctionDecider(d)
    raise TypeError(f"decider must have ask(requests) -> answers or be a function fn(state, question); got {type(d).__name__}")


# --- cached remote models --------------------------------------------------------------------
def post_json(url: str, body: dict, key: str | None, label: str, attempts: int = 8, timeout: float = 90, on_retry: Callable[[], None] | None = None) -> dict:
    """POST JSON with bearer auth, retrying busy and network errors with exponential backoff."""
    data = json.dumps(body).encode()
    delay = 1.0
    for attempt in range(attempts):
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        r = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode("utf-8", "replace")
            if e.code == 402:
                raise RuntimeError(f"{label} HTTP 402: the account has no credits left") from e
            # Workers AI behind OpenRouter now and then rejects a valid request with a 400 "AiError"; it passes when resent.
            upstream = e.code == 400 and "AiError" in msg and attempt < 3
            if e.code not in RETRY_CODES and not upstream:
                raise RuntimeError(f"{label} HTTP {e.code}: {msg}") from e
            log.warning("%s HTTP %d (attempt %d), retrying in %.0fs: %s", label, e.code, attempt + 1, delay, msg)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log.warning("%s network error (attempt %d), retrying in %.0fs: %s", label, attempt + 1, delay, e)
        if on_retry:
            on_retry()
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(f"{label} request failed after {attempts} attempts")


class CachedDecider:
    """Cache every answer by the SHA-256 of (model, request), dedupe, ask concurrently, replay offline.

    Subclasses implement `_call(req) -> compact answers`. With `offline=True` a cache miss raises CacheMiss.
    """

    label = "decider"

    def __init__(self, model: str, cache: str | Path | None = None, offline: bool = False, workers: int = 16):
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
            log.info("loaded %d cached %s answers from %s", len(self.store), self.label, self.path)

    def _call(self, req: dict) -> dict:
        raise NotImplementedError

    def _retried(self) -> None:
        with self.lock:
            self.stats["retries"] += 1

    def _post(self, url: str, body: dict, key_env: str | None) -> dict:
        """POST to the model; `key_env=None` sends no auth (a local server)."""
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key:
            raise RuntimeError(f"{key_env} is not set")
        out = post_json(url, body, key, self.label, on_retry=self._retried)
        with self.lock:
            self.stats["api_calls"] += 1
            u = out.get("usage") or {}
            self.stats["input_tokens"] += int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0)
            self.stats["cost_usd"] += float(u.get("cost", 0.0) or 0.0)
            self.models_seen.add(out.get("model", "?"))
        return out

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
        log.info("%s %s answered %d requests in %.1fs (cache hits so far %d, api calls %d)", self.label, self.model, len(reqs), time.time() - t0, self.stats["cache_hits"], self.stats["api_calls"])
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
        log.info("saved %d %s answers to %s", len(rows), self.label, self.path)


# --- any chat model --------------------------------------------------------------------------
SYSTEM = (
    "You are a decision model. Read `state` and answer every question in `questions`, judging only by each "
    "question's `instructions` and `criteria`. Reply with one JSON object and nothing else that maps every "
    'question id to its answer. For a question of type "noul": the probability, from 0 to 1, that its statement '
    'is true. For a question of type "choice": an object that maps every option name in its `criteria` to the '
    "probability that option is right; the probabilities sum to 1."
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_reply(text: str) -> dict:
    """Read the JSON object out of a chat reply, tolerating code fences and prose around it."""
    s = _FENCE.sub("", (text or "").strip())
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ValueError(f"reply has no JSON object: {text[:200]!r}") from None
        out = json.loads(s[a : b + 1])
    if not isinstance(out, dict):
        raise ValueError(f"reply is not a JSON object: {text[:200]!r}")
    return out


class LLM(CachedDecider):
    """Any chat model as a decision model, through an OpenAI-compatible /chat/completions endpoint.

    LLM("openai/gpt-5-mini")                                     # OpenRouter, OPENROUTER_API_KEY
    LLM("gpt-5-mini", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
    LLM("qwen3:8b", base_url="http://localhost:11434/v1", api_key_env=None)   # Ollama, no key

    The model is told the answer shape; loose replies (fenced JSON, an option name, percentages) are
    coerced, and a reply that still misses a question is asked again up to `repairs` times.
    """

    label = "LLM"

    def __init__(
        self,
        model: str,
        cache: str | Path | None = None,
        offline: bool = False,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str | None = "OPENROUTER_API_KEY",
        workers: int = 16,
        temperature: float = 0.0,
        json_mode: bool = True,
        repairs: int = 2,
    ):
        super().__init__(model, cache, offline, workers)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key_env, self.temperature, self.json_mode, self.repairs = api_key_env, temperature, json_mode, repairs

    def _call(self, req: dict) -> dict:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"state": req["state"], "questions": req["questions"]}, ensure_ascii=False)},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        err = None
        for attempt in range(self.repairs + 1):
            out = self._post(self.url, body, self.api_key_env)
            try:
                reply = out["choices"][0]["message"]["content"]
                return normalize([req], [parse_reply(reply)])[0]
            except (KeyError, IndexError, TypeError, ValueError) as e:
                err = e
                log.warning("LLM %s gave an unusable reply (attempt %d): %s", self.model, attempt + 1, e)
        raise RuntimeError(f"LLM {self.model} gave no usable answer after {self.repairs + 1} tries: {err}")


# --- typed decision models ------------------------------------------------------------------
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def _compact(answers: dict) -> dict:
    out = {}
    for qid, a in answers.items():
        if a.get("type") == "noul":
            out[qid] = round(float(a["noul"]), 4)
        else:
            out[qid] = {o: round(float(p), 4) for o, p in a["probabilities"].items()}
    return out


class Decisions(CachedDecider):
    """A typed decision model through OpenRouter's Decisions API, which answers typed questions natively (no prompt).

    Decisions("typesafe/jev-1.14")      # TypeSafe Jev
    Decisions("cloudflare/clef")        # Cloudflare Clef, open weights (Apache 2.0)
    Decisions("cloudflare/clef-flash")  # its 9B sibling
    """

    label = "Decisions"

    def __init__(self, model: str, cache: str | Path | None = None, offline: bool = False, workers: int = 16, url: str = DECISIONS_URL, api_key_env: str | None = "OPENROUTER_API_KEY"):
        super().__init__(model, cache, offline, workers)
        self.url, self.api_key_env = url, api_key_env

    def _call(self, req: dict) -> dict:
        sent, back = _wire(req)
        ans = _compact(self._post(self.url, {"model": self.model, **sent}, self.api_key_env)["answers"])
        return {qid: {back[qid].get(o, o): p for o, p in a.items()} if qid in back else a for qid, a in ans.items()}


def _wire(req: dict) -> tuple[dict, dict]:
    """Option names without "/": Cloudflare's Workers AI rejects a Choice with two such names (HTTP 400 AiError).

    The cache key is taken from the request as d-means wrote it, so this only changes what goes over the wire.
    Returns (request to send, {qid: {sent option: original option}}).
    """
    qs, back = {}, {}
    for qid, q in req["questions"].items():
        crit = q.get("criteria")
        if q.get("type") == "choice" and isinstance(crit, dict) and sum("/" in o for o in crit) > 1:
            names = {o: o.replace("/", "|") for o in crit}
            if len(set(names.values())) == len(names) and not set(names.values()) & (set(crit) - {o for o in crit if "/" in o}):
                qs[qid] = {**q, "criteria": {names[o]: v for o, v in crit.items()}}
                back[qid] = {n: o for o, n in names.items()}
                continue
        qs[qid] = q
    return ({**req, "questions": qs} if back else req), back
