"""M6: the browser version (web/tinygpt.js) against the Python code it was ported from.

Runs the JavaScript with Node.js (skipped if `node` is not installed) and compares, piece by piece:
text normalisation, the tokenizer, the model's logits, greedy generation (window restart included),
the decoding filters, the parse-tree functions, the parser and streaming.
"""
import json
import random
import shutil
import subprocess

import pytest
import torch

pytest.importorskip("safetensors")

from tinygpt.bpe import BPETokenizer  # noqa: E402
from tinygpt.bundle import save_bundle  # noqa: E402
from tinygpt.models import GPT, GPTConfig  # noqa: E402
from tinygpt.sampling import filter_logits  # noqa: E402
from tinygpt.serve.engine import GenerateParams, Parser, TextGenerator, word_spans  # noqa: E402
from tinygpt.text import normalize  # noqa: E402
from tinygpt.top import decoupled, from_model, parse_tree, slot_items, top_intent  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js to run web/tinygpt.js")
TOKENIZER = "tokenizer/tinygpt-indic-8k.json"
TRICKY = ["  नमस्ते\r\nदुनिया  ", "a\u00a0\u2003b\tc", "x\u200bz", "क्\u200dष", "\ufeffhello", "one\n\n\n\ntwo",
          "café", "\x00ctrl\x07", "क़", "it's HE'LL we'Ve", "mujhe 2 baje ka alarm, please!!",
          "₹500 का नोट 😀", "line sep", "x\u0085y", "महात्मा गांधी का जन्म १८६९ में हुआ", "  \n  ", ""]


def js(command, payload, tmp_path):
    f = tmp_path / f"{command}.json"
    f.write_text(json.dumps(payload, ensure_ascii=False))
    out = subprocess.run([NODE, "web/test/cli.mjs", command, str(f)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def random_texts(n=300, seed=0):
    rng = random.Random(seed)
    pools = ["कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह", "ािीुूृेैोौंःँ़्", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
             "0123456789०१२३४५६७८९", ".,!?;:'\"()[]-–—/₹%&@#…", " \u00a0\u2009\t\n\r\u3000", "\u200c\u200d\u0085\ufeff😀🙏"]
    return ["".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 40))) for _ in range(n)]


def peaky_gpt(block_size=16, seed=0) -> GPT:
    """A tiny random Llama-recipe GPT whose next-token choices are decisive (a scaled-up output
    layer), so greedy decoding can't be flipped by float rounding between Python and JavaScript."""
    torch.manual_seed(seed)
    m = GPT(GPTConfig(vocab_size=8192, block_size=block_size, d_model=32, n_layers=2, n_heads=2, dropout=0.0,
                      style="llama")).eval()
    with torch.no_grad():
        m.wte.weight.mul_(40)
    return m


# ------------------------------------------------------------------ text and tokenizer
def test_normalize_matches_python(tmp_path):
    texts = TRICKY + random_texts(100, seed=1)
    assert js("normalize", {"texts": texts}, tmp_path) == [normalize(t) for t in texts]


def test_tokenizer_matches_python(tmp_path):
    tok = BPETokenizer.load(TOKENIZER)
    texts = TRICKY + random_texts()
    assert js("tokenize", {"tokenizer": TOKENIZER, "texts": texts}, tmp_path) == [tok.encode_ordinary(t) for t in texts]


# ------------------------------------------------------------------ the model
def test_logits_match_pytorch(tmp_path):
    m = peaky_gpt()
    save_bundle(tmp_path / "b", m, TOKENIZER, {"name": "lm"})
    ids = torch.randint(0, 8192, (16,), generator=torch.Generator().manual_seed(1)).tolist()
    with torch.no_grad():
        expected = m(torch.tensor([ids]))[0][0]
    got = torch.tensor(js("logits", {"bundle": str(tmp_path / "b"), "ids": ids}, tmp_path))
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=1e-4)


def test_greedy_generation_matches_including_window_restart(tmp_path):
    """40 new tokens in a 16-token window: the cache restarts from the last half twice."""
    m = peaky_gpt()
    save_bundle(tmp_path / "b", m, TOKENIZER, {"name": "lm"})
    prompt = [8184, 5, 77, 1024, 300]
    expected = [int(t[0, 0]) for t in m.iter_tokens(torch.tensor([prompt]), 40, temperature=0)]
    assert js("greedy", {"bundle": str(tmp_path / "b"), "ids": prompt, "n": 40}, tmp_path) == expected


def test_filters_match_sampling_py(tmp_path):
    g = torch.Generator().manual_seed(3)
    cases = []
    for params in [dict(temperature=0.7, topK=5), dict(temperature=1.0, topP=0.9), dict(temperature=1.0, minP=0.1),
                   dict(temperature=0.8, topK=20, topP=0.8, minP=0.05, repetitionPenalty=1.3, history=[3, 3, 7, 40]),
                   dict(temperature=1.3, repetitionPenalty=1.5, history=[0, 1, 2])]:
        cases.append({"logits": (torch.randn(50, generator=g) * 3).tolist(), "params": params})
    got = js("filter", {"cases": cases}, tmp_path)
    for case, out in zip(cases, got):
        p = case["params"]
        expected = filter_logits(torch.tensor([case["logits"]]), p["temperature"], p.get("topK"), top_p=p.get("topP"),
                                 min_p=p.get("minP"), repetition_penalty=p.get("repetitionPenalty", 1.0),
                                 history=torch.tensor([p["history"]]) if "history" in p else None)[0]
        assert [v is None for v in out] == torch.isinf(expected).tolist()
        kept = ~torch.isinf(expected)
        torch.testing.assert_close(torch.tensor([v for v in out if v is not None]), expected[kept])


def test_streamed_text_matches_the_server_engine(tmp_path):
    m = peaky_gpt(block_size=64)
    save_bundle(tmp_path / "b", m, TOKENIZER, {"name": "lm"})
    params = GenerateParams(max_new_tokens=30, temperature=0)
    events = list(TextGenerator.from_bundle(tmp_path / "b").stream("भारत की", params))
    got = js("stream", {"bundle": str(tmp_path / "b"), "prompt": "भारत की",
                        "params": {"maxNewTokens": 30, "temperature": 0, "seed": 1}}, tmp_path)
    assert got["text"] == "".join(e.text for e in events[:-1])
    assert (got["done"]["tokens"], got["done"]["finishReason"]) == (events[-1].tokens, events[-1].finish_reason)


# ------------------------------------------------------------------ parsing
PARSES = ["[IN:CREATE_ALARM [SL:DATE_TIME kal subah 7 baje ] ka alarm laga do ]",
          "[IN:GET_WEATHER mujhe [SL:LOCATION Delhi ] ka weather batao ]",
          "[IN:GET_EVENT [SL:LOCATION [IN:GET_LOCATION [SL:CATEGORY_LOCATION Library ] ] ] [SL:LOCATION Cedar Park ] me ]",
          "[IN:PAUSE_TIMER timer roko ]", "[IN:X [SL:Y ] ]", "[IN:A w1 [SL:B w2 [IN:C [SL:D w3 w4 ] ] ] w5 ]",
          "[SL:NOT_ROOT x ]", "[IN:A [IN:B x ] ]", "[IN:A x", "word [IN:A ]", "[IN:A ] [IN:B ]", "[IN: x ]", ""]


def test_tree_functions_match_top_py(tmp_path):
    got = js("trees", {"parses": PARSES}, tmp_path)
    for p, out in zip(PARSES, got):
        try:
            tree = parse_tree(p)
        except ValueError:
            assert out == {"error": True}, p
            continue
        assert out["decoupled"] == decoupled(p)
        assert [tuple(s) for s in out["slots"]] == slot_items(tree)
        assert out["intent"] == top_intent(p)
        assert out["spans"] == word_spans(tree)
        assert out["upper"] == from_model(p.lower())


def test_parser_matches_the_server_parser(tmp_path):
    save_bundle(tmp_path / "p", peaky_gpt(block_size=64, seed=1), TOKENIZER,
                {"name": "hinglish-top", "labels": "lower", "target": "annotated"})
    texts = ["kal subah 7 baje ka alarm laga do", "timer roko", "mujhe  Delhi ka\tweather batao", "", "q " * 100]
    parser = Parser.from_bundle(tmp_path / "p")
    # one request at a time, as the browser parses: in a tiny 64-token window a batch's longest
    # request would shorten everyone's token budget (with the real 256-token window it can't)
    expected = [parser.parse([t])[0] for t in texts]
    got = js("parse", {"bundle": str(tmp_path / "p"), "texts": texts}, tmp_path)
    for e, g in zip(expected, got):
        assert (g["text"], g["written"], g["parse"], g["valid"], g["tokens"]) == (e.text, e.written, e.parse, e.valid, e.tokens)
        assert (g["error"] is None) == (e.error is None)
