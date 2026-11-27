"""Decision models other than Jev: any chat LLM, a plain function, and the checks DMeans applies."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from test_dmeans import corpus  # noqa: F401  (fixture)

from dmeans import DMeans
from dmeans import deciders as decmod
from dmeans.deciders import LLM, CacheMiss, FunctionDecider, coerce
from dmeans.metrics import accuracy

REQ = {
    "state": {"text": "The striker scored twice."},
    "questions": {
        "group": {"type": "choice", "instructions": "Which group?", "criteria": {"sport": "games", "politics": "votes"}},
        "same": {"type": "noul", "instructions": "Is it sport?"},
    },
}


class Chat(BaseHTTPRequestHandler):
    calls: list = []
    replies: list = []  # queued raw reply texts; default is a clean JSON answer

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Chat.calls.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        content = Chat.replies.pop(0) if Chat.replies else json.dumps({"group": {"sport": 3, "politics": 1}, "same": 0.8})
        out = {"model": body["model"], "choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 120, "cost": 1e-5}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def chat(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Chat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("TEST_LLM_KEY", "llm-key")
    monkeypatch.setattr(decmod.time, "sleep", lambda s: None)
    Chat.calls, Chat.replies = [], []
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_llm_answers_in_the_compact_shape(chat):
    llm = LLM("openai/gpt-5-mini", base_url=chat, api_key_env="TEST_LLM_KEY")
    ans = llm.ask([REQ])[0]
    call = Chat.calls[0]
    assert call["path"] == "/v1/chat/completions" and call["auth"] == "Bearer llm-key"
    assert call["body"]["model"] == "openai/gpt-5-mini" and call["body"]["temperature"] == 0.0
    sent = json.loads(call["body"]["messages"][1]["content"])
    assert sent == {"state": REQ["state"], "questions": REQ["questions"]}
    assert ans["same"] == pytest.approx(0.8)
    assert ans["group"] == {"sport": 0.75, "politics": 0.25}
    assert llm.stats["api_calls"] == 1 and llm.stats["input_tokens"] == 120


def test_llm_repairs_loose_and_broken_replies(chat):
    Chat.replies = [
        "I think so.",  # no JSON: asked again
        '```json\n{"group": "politics", "same": "87%"}\n```',
    ]
    llm = LLM("local-model", base_url=chat, api_key_env=None)
    ans = llm.ask([REQ])[0]
    assert len(Chat.calls) == 2 and Chat.calls[0]["auth"] is None
    assert ans == {"group": {"sport": 0.0, "politics": 1.0}, "same": pytest.approx(0.87)}
    Chat.replies = ['{"group": {"sport": 1}}'] * 3  # never answers `same`
    with pytest.raises(RuntimeError, match="no usable answer"):
        LLM("local-model", base_url=chat, api_key_env=None, repairs=2).ask([{**REQ, "state": {"text": "x"}}])


def test_llm_cache_answers_once_and_replays_offline(chat, tmp_path):
    path = tmp_path / "llm.jsonl.gz"
    live = LLM("anthropic/claude-haiku-5-5", cache=path, base_url=chat, api_key_env="TEST_LLM_KEY")
    live.ask([REQ, REQ])
    live.save()
    assert len(Chat.calls) == 1
    offline = LLM("anthropic/claude-haiku-5-5", cache=path, offline=True)
    assert offline.ask([REQ])[0]["same"] == pytest.approx(0.8)
    assert len(Chat.calls) == 1 and offline.stats["api_calls"] == 0
    with pytest.raises(CacheMiss):
        LLM("another/model", cache=path, offline=True).ask([REQ])  # the key includes the model


def test_function_decider_drives_dmeans(corpus):  # noqa: F811
    x, texts, y = corpus
    label = dict(zip(texts, y))

    def card_class(card):
        return np.bincount([label[t] for t in card["examples"]], minlength=4).argmax()

    def judge(state, question):
        if question["type"] == "noul":
            return card_class(state["group_a"]) == card_class(state["group_b"])  # a bool is fine
        truth = label[state["text"]]
        return next((n for n, c in question["criteria"].items() if card_class(c) == truth), next(iter(question["criteria"])))

    dm = DMeans(4, judge, seed=0)  # default k-means start asks only point questions
    assert isinstance(dm.decider, FunctionDecider)
    dm.fit(x, texts)
    assert dm.report["pair_decisions"] == 0 and dm.report["point_decisions"] > 0

    # The corpus interleaves classes in space, so only the merge path can recover them.
    dm = DMeans(4, judge, over=3, seed=0, start="merge")
    lab = dm.fit(x, texts)
    assert dm.report["pair_decisions"] > 0 and dm.report["point_decisions"] > 0
    assert accuracy(y, lab) > 0.97


def test_rejects_bad_deciders_and_answers(corpus):  # noqa: F811
    x, texts, _ = corpus
    with pytest.raises(TypeError, match="decider must have ask"):
        DMeans(4, "jev")

    class Forgetful:
        def ask(self, reqs):
            return [{} for _ in reqs]

    with pytest.raises(ValueError, match="no answer to question 'group'"):
        DMeans(4, Forgetful(), seed=0).fit(x, texts)
    with pytest.raises(ValueError, match="no answer to question 'same'"):
        DMeans(4, Forgetful(), seed=0, start="merge").fit(x, texts)
    with pytest.raises(ValueError, match="names no option"):
        coerce(REQ["questions"]["group"], "chess", "group")
