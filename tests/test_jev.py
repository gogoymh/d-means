"""Jev client against a local stand-in for OpenRouter's Decisions API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dmeans import jev as jevmod
from dmeans.jev import CacheMiss, Jev


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
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"busy"}}')
            return
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.87654}
            else:
                opts = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": opts[0], "probabilities": {o: (0.9 if i == 0 else 0.1 / (len(opts) - 1)) for i, o in enumerate(opts)}, "confidence": 0.8}
        out = {"model": "typesafe/jev-1.14-20261014", "answers": answers, "usage": {"input_tokens": 100, "cost": 4.2e-6}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(jevmod, "URL", f"http://127.0.0.1:{srv.server_port}/api/alpha/decisions")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(jevmod.time, "sleep", lambda s: None)
    Fake.calls, Fake.fail_next = [], []
    yield srv
    srv.shutdown()


REQ = {
    "state": {"text": "The striker scored twice."},
    "questions": {
        "group": {"type": "choice", "instructions": "Which group?", "criteria": {"sport": "games", "politics": "votes"}},
        "same": {"type": "noul", "instructions": "Is it sport?"},
    },
}


def test_request_shape_and_compact_answers(server):
    ans = Jev().ask([REQ])[0]
    call = Fake.calls[0]
    assert call["auth"] == "Bearer test-key"
    assert call["body"]["model"] == "typesafe/jev-1.14"
    assert call["body"]["state"] == REQ["state"] and call["body"]["questions"] == REQ["questions"]
    assert ans["same"] == pytest.approx(0.8765)
    assert ans["group"] == {"sport": 0.9, "politics": 0.1}


def test_cache_answers_once_and_replays_offline(server, tmp_path):
    path = tmp_path / "answers.jsonl.gz"
    live = Jev(path)
    live.ask([REQ, REQ])
    live.save()
    assert len(Fake.calls) == 1  # the duplicate came from the cache
    offline = Jev(path, offline=True)
    assert offline.ask([REQ])[0]["same"] == pytest.approx(0.8765)
    assert len(Fake.calls) == 1 and offline.stats["api_calls"] == 0
    with pytest.raises(CacheMiss):
        offline.ask([{**REQ, "state": {"text": "something new"}}])


def test_retries_busy_and_stops_on_no_credit(server):
    Fake.fail_next = [429, 503]
    j = Jev()
    assert j.ask([REQ])[0]["same"] == pytest.approx(0.8765)
    assert j.stats["retries"] == 2
    Fake.fail_next = [402]
    with pytest.raises(RuntimeError, match="no credits"):
        Jev().ask([{**REQ, "state": {"text": "other"}}])
