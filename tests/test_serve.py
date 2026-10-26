"""M6: bundles, the engines, the worker threads, the HTTP API and the demo — with tiny random models."""
import json
import threading
import time

import pytest
import torch

pytest.importorskip("safetensors")

from tinygpt.bundle import load_bundle, read_config, save_bundle  # noqa: E402
from tinygpt.models import GPT, GPTConfig  # noqa: E402
from tinygpt.sampling import stream as reference_stream  # noqa: E402
from tinygpt.serve.engine import (Done, GenerateParams, Parser, ParseResult, Piece, TextGenerator,  # noqa: E402
                                  greedy_batch, word_spans)
from tinygpt.serve.worker import Busy, Error, GenerateWorker, Job, ParseWorker, SyncChannel  # noqa: E402
from tinygpt.top import parse_tree  # noqa: E402

TOKENIZER = "tokenizer/tinygpt-indic-8k.json"
PARSER_TASK = {"name": "hinglish-top", "labels": "lower", "target": "annotated"}


def tiny_gpt(seed=0, **kw) -> GPT:
    torch.manual_seed(seed)
    cfg = dict(vocab_size=8192, block_size=64, d_model=32, n_layers=2, n_heads=2, dropout=0.0, style="llama")
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    save_bundle(root / "lm", tiny_gpt(0), TOKENIZER, {"name": "lm"}, scores={"val_bpb": {"all": 9.9}})
    save_bundle(root / "parser", tiny_gpt(1), TOKENIZER, PARSER_TASK,
                scores={"test": {"exact_match": 0.41, "intent_acc": 0.84}})
    return root


@pytest.fixture(scope="module")
def generator(bundles):
    return TextGenerator.from_bundle(bundles / "lm")


@pytest.fixture(scope="module")
def parser(bundles):
    return Parser.from_bundle(bundles / "parser", max_new_tokens=12)


# ------------------------------------------------------------------ bundles
def test_bundle_round_trip(tmp_path):
    model = tiny_gpt(3)
    save_bundle(tmp_path / "b", model, TOKENIZER, {"name": "lm"})
    loaded, tok, config = load_bundle(tmp_path / "b")
    x = torch.randint(0, 8192, (2, 16))
    assert torch.equal(model(x)[0], loaded(x)[0])
    assert loaded.head.weight is loaded.wte.weight                  # still one tied matrix
    assert config["params"] == model.num_params() and tok.vocab_size == 8192
    assert (tmp_path / "b" / "model.safetensors").stat().st_size < model.num_params() * 4 + 10_000   # stored once
    assert (tmp_path / "b" / "model.safetensors").stat().st_mode & 0o777 == 0o644      # anyone may read it


def test_bundle_checksums_catch_a_changed_file(tmp_path):
    save_bundle(tmp_path / "b", tiny_gpt(), TOKENIZER, {"name": "lm"})
    path = tmp_path / "b" / "tokenizer.json"
    path.write_text(path.read_text().replace('"version": 1', '"version": 1 '))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "b")
    load_bundle(tmp_path / "b", verify=False)


def test_bundle_from_the_future_is_refused(tmp_path):
    save_bundle(tmp_path / "b", tiny_gpt(), TOKENIZER, {"name": "lm"})
    config = json.loads((tmp_path / "b" / "config.json").read_text())
    config["version"] = 99
    (tmp_path / "b" / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="version"):
        read_config(tmp_path / "b")


# ------------------------------------------------------------------ engines
def test_stream_matches_m4_streaming(generator):
    params = GenerateParams(max_new_tokens=20, temperature=1.0, top_k=None, seed=7)
    events = list(generator.stream("भारत की", params))
    assert all(isinstance(e, Piece) for e in events[:-1]) and isinstance(events[-1], Done)
    expected = "".join(reference_stream(generator.model, generator.tok, "भारत की", 20, temperature=1.0,
                                        generator=torch.Generator().manual_seed(7)))
    assert "".join(e.text for e in events[:-1]) == expected
    done = events[-1]
    assert done.seed == 7 and done.finish_reason in ("stop", "length") and done.tokens <= 20
    assert done.first_token_seconds is not None and done.seconds >= done.first_token_seconds


def test_same_seed_same_text_and_random_seed_is_reported(generator):
    def text(params):
        events = list(generator.stream("yaar", params))
        return "".join(e.text for e in events[:-1]), events[-1]
    a, _ = text(GenerateParams(max_new_tokens=15, seed=3))
    b, _ = text(GenerateParams(max_new_tokens=15, seed=3))
    assert a == b
    c, done = text(GenerateParams(max_new_tokens=15))
    assert c == text(GenerateParams(max_new_tokens=15, seed=done.seed))[0]


def test_cancelling_stops_generation(generator):
    n = {"pieces": 0}

    def cancelled():
        n["pieces"] += 1
        return n["pieces"] >= 3
    done = list(generator.stream("भारत", GenerateParams(max_new_tokens=50, temperature=1.0, top_k=None, seed=1),
                                 cancelled=cancelled))[-1]
    assert done.finish_reason == "cancelled" and done.tokens == 3


def test_batched_parsing_equals_one_at_a_time(parser):
    texts = ["kal subah 7 baje ka alarm laga do", "timer roko", "mujhe Delhi ka weather batao", "gaana bajao"]
    together = parser.parse(texts)
    assert [r.batch_size for r in together] == [4] * 4
    for t, r in zip(texts, together):
        alone = parser.parse([t])[0]
        assert (alone.written, alone.tokens) == (r.written, r.tokens)


def test_greedy_batch_needs_no_equal_lengths(parser):
    tok, m = parser.tok, parser.model
    prompts = [[tok.eot_id, 5, 6], [tok.eot_id, 7, 8, 9, 10, 11], [tok.eot_id]]
    outs = greedy_batch(m, prompts, tok.eot_id, parser.pad_id, 8)
    for p, out in zip(prompts, outs):
        assert out == greedy_batch(m, [p], tok.eot_id, parser.pad_id, 8)[0]


def test_bad_request_fails_alone(parser):
    results = parser.parse(["", "timer roko", "bahut " * 80])
    assert results[0].error == "the request is empty"
    assert "limit" in results[2].error
    assert results[1].error is None and results[1].text == "timer roko"


def test_result_from_a_written_tree(parser):
    out = parser.tok.encode_ordinary(" [in:create_alarm [sl:date_time kal subah 7 baje ] ka alarm laga do ]")
    r = parser._result("kal subah 7 baje ka alarm laga do", out, 1)
    assert r.valid and r.intent == "CREATE_ALARM"
    assert r.parse == "[IN:CREATE_ALARM [SL:DATE_TIME kal subah 7 baje ] ]"
    assert r.slots == [{"label": "DATE_TIME", "text": "kal subah 7 baje"}]
    assert r.spans == [{"text": "kal subah 7 baje", "label": "DATE_TIME"}, {"text": "ka alarm laga do", "label": None}]
    bad = parser._result("x", parser.tok.encode_ordinary(" [in:oops [sl:x"), 1)
    assert not bad.valid and bad.slots == [] and bad.intent is None


def test_word_spans_use_the_innermost_slot():
    tree = parse_tree("[IN:A w1 [SL:B w2 [IN:C [SL:D w3 w4 ] ] ] w5 ]")
    assert word_spans(tree) == [{"text": "w1", "label": None}, {"text": "w2", "label": "B"},
                                {"text": "w3 w4", "label": "D"}, {"text": "w5", "label": None}]


# ------------------------------------------------------------------ workers
class GatedParser:
    """A parser whose first call blocks until released: lets the test fill the queue meanwhile."""

    def __init__(self):
        self.gate, self.started, self.batches = threading.Event(), threading.Event(), []

    def parse(self, texts):
        self.started.set()
        self.gate.wait(5)
        self.batches.append(list(texts))
        return [ParseResult(t, "", "", False, None, batch_size=len(texts)) for t in texts]


def test_parse_worker_batches_what_waits():
    p = GatedParser()
    w = ParseWorker(p, max_queue=10, max_batch=3).start()
    jobs = [w.submit(Job("first", SyncChannel()))]
    assert p.started.wait(5)                        # "first" is running alone
    jobs += [w.submit(Job(f"r{i}", SyncChannel())) for i in range(4)]
    p.gate.set()
    results = [j.channel.get(timeout=5) for j in jobs]
    w.stop()
    assert p.batches == [["first"], ["r0", "r1", "r2"], ["r3"]]
    assert [r.text for r in results] == ["first", "r0", "r1", "r2", "r3"]


def test_full_queue_is_busy_and_cancelled_jobs_are_skipped():
    p = GatedParser()
    w = ParseWorker(p, max_queue=2, max_batch=8).start()
    w.submit(Job("running", SyncChannel()))
    assert p.started.wait(5)
    waiting = [w.submit(Job(f"w{i}", SyncChannel())) for i in range(2)]
    with pytest.raises(Busy):
        w.submit(Job("one too many", SyncChannel()))
    waiting[0].cancel.set()                         # its client left while it waited
    p.gate.set()
    assert waiting[1].channel.get(timeout=5).text == "w1"
    w.stop()
    assert p.batches == [["running"], ["w1"]]


def test_worker_reports_errors_and_keeps_running():
    class Broken:
        calls = 0

        def parse(self, texts):
            Broken.calls += 1
            if Broken.calls == 1:
                raise RuntimeError("boom")
            return [ParseResult(t, "", "", False, None) for t in texts]
    w = ParseWorker(Broken(), max_queue=4).start()
    first = w.submit(Job("a", SyncChannel())).channel.get(timeout=5)
    second = w.submit(Job("b", SyncChannel())).channel.get(timeout=5)
    w.stop()
    assert isinstance(first, Error) and "boom" in first.message
    assert second.text == "b"


def test_generate_worker_streams_and_cancels(generator):
    w = GenerateWorker(generator, max_queue=2).start()
    job = w.submit(Job(("भारत", GenerateParams(max_new_tokens=40, temperature=1.0, top_k=None, seed=2)),
                       SyncChannel()))
    first = job.channel.get(timeout=10)
    job.cancel.set()
    events = [first]
    while not isinstance(events[-1], Done):
        events.append(job.channel.get(timeout=10))
    w.stop()
    assert isinstance(first, Piece) and events[-1].finish_reason in ("cancelled", "stop")


# ------------------------------------------------------------------ the HTTP API
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tinygpt.serve.api import Settings, create_app  # noqa: E402


@pytest.fixture()
def client(generator, parser):
    app = create_app(Settings(ui=False, max_queue_generate=2), generator=generator, parser=parser)
    with TestClient(app) as c:
        yield c


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("TINYGPT_MAX_BATCH", "1")
    monkeypatch.setenv("TINYGPT_MAX_WAIT_MS", "2.5")
    monkeypatch.setenv("TINYGPT_UI", "0")
    monkeypatch.setenv("TINYGPT_LM", "/models/lm")
    s = Settings.from_env()
    assert (s.max_batch, s.max_wait_ms, s.ui, s.lm, s.parser) == (1, 2.5, False, "/models/lm", "models/parser")


def test_health_info_and_request_id(client):
    r = client.get("/health", headers={"x-request-id": "abc-123"})
    assert r.json() == {"status": "ok"} and r.headers["x-request-id"] == "abc-123"
    assert len(client.get("/health").headers["x-request-id"]) == 16          # made up when not given
    info = client.get("/v1/info").json()
    assert info["models"]["parse"]["task"] == PARSER_TASK and info["limits"]["max_batch"] == 16


def test_generate_json(client):
    body = {"prompt": "भारत की", "max_new_tokens": 10, "seed": 4}
    a, b = client.post("/v1/generate", json=body).json(), client.post("/v1/generate", json=body).json()
    assert a["text"] == b["text"] and a["seed"] == 4 and a["tokens"] <= 10
    assert a["finish_reason"] in ("stop", "length") and a["queue_seconds"] >= 0


@pytest.mark.parametrize("bad", [{"temperature": 5}, {"max_new_tokens": 0}, {"top_p": 0}, {"tempreature": 1},
                                 {"prompt": "x" * 2001}])
def test_generate_rejects_bad_settings(client, bad):
    assert client.post("/v1/generate", json={"prompt": "a", **bad}).status_code == 422


def test_generate_stream_is_server_sent_events(client):
    body = {"prompt": "yaar", "max_new_tokens": 8, "seed": 5}
    with client.stream("POST", "/v1/generate/stream", json=body) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        lines = list(r.iter_lines())
    events = [line[7:] for line in lines if line.startswith("event: ")]
    data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    assert events[-1] == "done" and set(events[:-1]) <= {"token"}
    text = "".join(d["text"] for d, e in zip(data, events) if e == "token")
    assert text == client.post("/v1/generate", json=body).json()["text"]
    assert data[-1]["tokens"] <= 8 and data[-1]["seed"] == 5


def test_full_queue_returns_503(client, monkeypatch):
    gate = threading.Event()
    worker = client.app.state.lm

    def slow_stream(prompt, params, cancelled=lambda: False):
        gate.wait(5)
        yield Done(0, "stop", 0, 0.0, None)
    monkeypatch.setattr(worker.generator, "stream", slow_stream)
    jobs = [worker.submit(Job(("a", GenerateParams()), SyncChannel())) for _ in range(2)]
    time.sleep(0.2)                                  # the worker takes one; one waits
    jobs.append(worker.submit(Job(("b", GenerateParams()), SyncChannel())))
    for path in ("/v1/generate", "/v1/generate/stream"):
        r = client.post(path, json={"prompt": "c"})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
    gate.set()
    for j in jobs:
        assert isinstance(j.channel.get(timeout=5), Done)


def test_parse_endpoint(client):
    r = client.post("/v1/parse", json={"text": "kal subah 7 baje ka alarm laga do"}).json()
    assert r["text"] == "kal subah 7 baje ka alarm laga do" and r["batch_size"] == 1
    assert set(r) >= {"written", "parse", "valid", "intent", "slots", "spans", "tokens", "queue_seconds"}
    assert client.post("/v1/parse", json={"text": "q " * 100}).status_code == 422          # 202 tokens > 64
    assert client.post("/v1/parse", json={"text": "   "}).status_code == 422
    assert client.post("/v1/parse", json={"text": "a", "extra": 1}).status_code == 422


def test_metrics(client):
    client.post("/v1/parse", json={"text": "timer roko"})
    client.post("/v1/generate", json={"prompt": "a", "max_new_tokens": 3})
    text = client.get("/metrics").text
    assert 'tinygpt_requests_total{endpoint="/v1/parse",status="200"} 1.0' in text
    assert "tinygpt_parse_batch_size_count 1.0" in text
    assert 'tinygpt_waiting_jobs{worker="generate"} 0.0' in text
    assert "tinygpt_generated_tokens_total" in text


# ------------------------------------------------------------------ the demo
def test_demo_is_mounted(generator, parser):
    pytest.importorskip("gradio")
    app = create_app(Settings(ui=True), generator=generator, parser=parser)
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert c.get("/health").json() == {"status": "ok"}            # the API still wins over the UI


def test_ui_helpers():
    pytest.importorskip("gradio")
    from tinygpt.serve.ui import params_from_ui, parse_outputs, stats_line
    p = params_from_ui(64, 0.7, 0, 1.0, 0.0, 1.0, -1)
    assert (p.top_k, p.top_p, p.min_p, p.seed) == (None, None, None, None)
    assert params_from_ui(64, 0.7, 40, 0.9, 0.1, 1.2, 5).seed == 5
    assert "seed 9" in stats_line(Done(10, "length", 9, 1.0, 0.1))
    ok = ParseResult("a", "[IN:X [SL:Y a ] ]", "[IN:X [SL:Y a ] ]", True, "X", [{"label": "Y", "text": "a"}],
                     [{"text": "a", "label": "Y"}])
    spans, summary, written = parse_outputs(ok)
    assert spans == [("a", "Y")] and "`X`" in summary and written == ok.written
    assert parse_outputs(ParseResult("", "", "", False, None, error="the request is empty"))[1].endswith("empty")
