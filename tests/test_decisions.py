"""Decisions client (Jev, Clef, ...) against a local stand-in for OpenRouter's Decisions API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dmeans import deciders as mod
from dmeans.deciders import CacheMiss, Decisions


class Fake(BaseHTTPRequestHandler):
    calls: list = []
    fail_next: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.calls.append({"auth": self.headers["Authorization"], "body": body})
        if Fake.fail_next:
            code = Fake.fail_next.pop(0)
            msg = {
                400: b'{"error":{"message":"HTTP 400: AiError: Bad input"}}',
                502: b'{"error":{"message":"OpenAI refused to answer question \\"group\\"","code":502}}',
                "blocked": b'{"error":{"message":"HTTP 400: Input blocked: The prompt contains sensitive words that violate Google\'s policy."}}',
            }
            self.send_response(400 if code == "blocked" else code)
            self.end_headers()
            self.wfile.write(msg.get(code, b'{"error":{"message":"busy"}}'))
            return
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.87654}
            else:
                opts = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": opts[0], "probabilities": {o: (0.9 if i == 0 else 0.1 / (len(opts) - 1)) for i, o in enumerate(opts)}, "confidence": 0.8}
        out = {"model": body["model"], "answers": answers, "usage": {"input_tokens": 100, "cost": 4.2e-6}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    Fake.calls, Fake.fail_next = [], []
    url = f"http://127.0.0.1:{srv.server_port}/api/alpha/decisions"
    yield lambda model="typesafe/jev-1.14", **kw: Decisions(model, url=url, **kw)
    srv.shutdown()


REQ = {
    "state": {"text": "The striker scored twice."},
    "questions": {
        "group": {"type": "choice", "instructions": "Which group?", "criteria": {"sport": "games", "politics": "votes"}},
        "same": {"type": "noul", "instructions": "Is it sport?"},
    },
}


@pytest.mark.parametrize("model", ["typesafe/jev-1.14", "cloudflare/clef", "cloudflare/clef-flash"])
def test_request_shape_and_compact_answers(server, model):
    ans = server(model).ask([REQ])[0]
    call = Fake.calls[0]
    assert call["auth"] == "Bearer test-key"
    assert call["body"]["model"] == model
    assert call["body"]["state"] == REQ["state"] and call["body"]["questions"] == REQ["questions"]
    assert ans["same"] == pytest.approx(0.8765)
    assert ans["group"] == {"sport": 0.9, "politics": 0.1}


def test_cache_answers_once_and_replays_offline(server, tmp_path):
    path = tmp_path / "answers.jsonl.gz"
    live = server("cloudflare/clef", cache=path)
    live.ask([REQ, REQ])
    live.save()
    assert len(Fake.calls) == 1  # the duplicate came from the cache
    offline = server("cloudflare/clef", cache=path, offline=True)
    assert offline.ask([REQ])[0]["same"] == pytest.approx(0.8765)
    assert len(Fake.calls) == 1 and offline.stats["api_calls"] == 0
    with pytest.raises(CacheMiss):
        offline.ask([{**REQ, "state": {"text": "something new"}}])
    with pytest.raises(CacheMiss):  # answers are keyed by model: Jev cannot replay Clef's
        server("typesafe/jev-1.14", cache=path, offline=True).ask([REQ])


def test_retries_busy_and_stops_on_no_credit(server):
    Fake.fail_next = [429, 503, 400]  # 400 AiError: a transient Workers AI rejection
    j = server("cloudflare/clef")
    assert j.ask([REQ])[0]["same"] == pytest.approx(0.8765)
    assert j.stats["retries"] == 3
    Fake.fail_next = [402]
    with pytest.raises(RuntimeError, match="no credits"):
        server().ask([{**REQ, "state": {"text": "other"}}])


def test_model_is_required_and_named_nowhere_else():
    with pytest.raises(TypeError):
        Decisions()  # no default model: the caller names it


def test_slash_option_names_go_over_the_wire_without_slashes(server):
    req = {"state": {"text": "x"}, "questions": {"group": {"type": "choice", "instructions": "?", "criteria": {"a / b": "one", "c / d": "two"}}}}
    ans = server("cloudflare/clef").ask([req])[0]
    assert list(Fake.calls[0]["body"]["questions"]["group"]["criteria"]) == ["a | b", "c | d"]
    assert ans["group"] == {"a / b": 0.9, "c / d": 0.1}
    assert req["questions"]["group"]["criteria"] == {"a / b": "one", "c / d": "two"}  # the caller's request is untouched


@pytest.mark.parametrize("refusal", [502, "blocked"])  # OpenAI refusal, Google input block
def test_refusal_gets_a_neutral_answer_once_and_replays(server, tmp_path, refusal):
    path = tmp_path / "answers.jsonl.gz"
    Fake.fail_next = [refusal]
    live = server("openai/gpt-6-luna-decisions", cache=path)
    ans = live.ask([REQ])[0]
    assert ans == {"group": {"sport": 0.5, "politics": 0.5}, "same": 0.5}
    assert len(Fake.calls) == 1 and live.stats["retries"] == 0  # a refusal is not retried
    live.save()
    offline = server("openai/gpt-6-luna-decisions", cache=path, offline=True)
    assert offline.ask([REQ])[0] == ans and len(offline.refused) == 1
