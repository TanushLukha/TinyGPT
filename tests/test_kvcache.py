import pytest
import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.models import GPT, GPTConfig, LSTMConfig, LSTMLM
from tinygpt.sampling import complete, stream

CASES = [("gpt2", "sdpa"), ("gpt2", "manual"), ("llama", "sdpa"), ("llama", "manual")]


def tiny(style="gpt2", attn="sdpa", **kw) -> GPT:
    torch.manual_seed(0)
    cfg = dict(vocab_size=60, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0, style=style, attn=attn)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


@pytest.mark.parametrize("style,attn", CASES)
def test_cached_logits_equal_full_forward(style, attn):
    """Prefill 5 tokens, then feed one token at a time: every logit must match one full pass."""
    m = tiny(style, attn)
    idx = torch.randint(0, 60, (2, 16))
    full, _ = m(idx)
    cache = m.new_cache(batch=2)
    parts = [m(idx[:, :5], cache=cache)[0]]
    for t in range(5, 16):
        parts.append(m(idx[:, t:t + 1], cache=cache)[0])
    assert cache.pos == 16
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("style,attn", CASES)
def test_generation_with_and_without_cache_agree(style, attn):
    m = tiny(style, attn)
    prompt = torch.randint(0, 60, (1, 4))
    for kwargs in (dict(temperature=0), dict(temperature=1.0, top_k=20)):
        a = m.generate(prompt, 12, use_cache=True, generator=torch.Generator().manual_seed(3), **kwargs)
        b = m.generate(prompt, 12, use_cache=False, generator=torch.Generator().manual_seed(3), **kwargs)
        assert torch.equal(a, b), kwargs


@pytest.mark.parametrize("style", ["gpt2", "llama"])
def test_generation_past_the_context_window(style):
    """Longer than block_size: the cache restarts from the last half of the text instead of failing."""
    m = tiny(style)
    out = m.generate(torch.randint(0, 60, (1, 3)), 50, temperature=0)
    assert out.shape == (1, 53)


def test_batched_generation_equals_one_by_one():
    m = tiny("llama")
    prompts = torch.randint(0, 60, (3, 5))
    together = m.generate(prompts, 8, temperature=0)
    for i in range(3):
        assert torch.equal(together[i:i + 1], m.generate(prompts[i:i + 1], 8, temperature=0))


def left_pad(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(map(len, rows))
    return (torch.tensor([[pad_id] * (T - len(r)) + r for r in rows]),
            torch.tensor([T - len(r) for r in rows]))


@pytest.mark.parametrize("style,attn", CASES)
def test_padded_batch_computes_what_each_row_computes_alone(style, attn):
    """Prompts of different lengths in one left-padded batch: every row's logits — after the prompt
    and after each new token — equal that row's logits without padding."""
    m = tiny(style, attn)
    torch.manual_seed(1)
    rows = [torch.randint(0, 59, (n,)).tolist() for n in (3, 7, 5)]
    x, pad = left_pad(rows, pad_id=59)
    cache = m.new_cache(batch=3, pad=pad)
    batched = [m(x, cache=cache)[0][:, -1]]
    nxt = torch.randint(0, 59, (3, 4))
    for t in range(4):
        batched.append(m(nxt[:, t:t + 1], cache=cache)[0][:, -1])
    for i, r in enumerate(rows):
        alone_cache = m.new_cache()
        alone = [m(torch.tensor([r]), cache=alone_cache)[0][:, -1]]
        for t in range(4):
            alone.append(m(nxt[i:i + 1, t:t + 1], cache=alone_cache)[0][:, -1])
        for b, a in zip(batched, alone):
            assert torch.isfinite(b[i]).all()
            torch.testing.assert_close(b[i:i + 1], a, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("style", ["gpt2", "llama"])
def test_padded_greedy_generation_equals_one_by_one(style):
    m = tiny(style)
    torch.manual_seed(2)
    rows = [torch.randint(0, 59, (n,)).tolist() for n in (2, 6, 4, 6)]
    x, pad = left_pad(rows, pad_id=59)
    together = torch.cat(list(m.iter_tokens(x, 8, temperature=0, pad=pad)), dim=1)
    for i, r in enumerate(rows):
        alone = torch.cat(list(m.iter_tokens(torch.tensor([r]), 8, temperature=0)), dim=1)
        assert torch.equal(together[i:i + 1], alone)


def test_padded_batch_limits():
    m = tiny("llama")                                    # block_size 16
    x, pad = left_pad([[1, 2, 3], [4]], pad_id=59)
    assert len(list(m.iter_tokens(x, 14, temperature=0, pad=pad))) == 14     # 3 + 14 - 1 = 16 fits
    with pytest.raises(ValueError):
        next(m.iter_tokens(x, 15, temperature=0, pad=pad))                   # would need a restart
    with pytest.raises(ValueError):
        next(m.iter_tokens(x, 4, temperature=0, pad=pad, use_cache=False))


def test_cache_too_long_is_an_error():
    m = tiny()
    cache = m.new_cache()
    m(torch.zeros(1, 10, dtype=torch.long), cache=cache)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 7, dtype=torch.long), cache=cache)


def test_cache_size():
    m = GPT(GPTConfig())                                 # 6 layers x 6 heads x 256 positions x 64 dims
    assert m.new_cache().nbytes() == 2 * 6 * 6 * 256 * 64 * 4      # keys + values, fp32: 4.7 MB


@pytest.fixture(scope="module")
def tok():
    text = ["भारत की राजधानी नई दिल्ली है। हिन्दी भाषा बहुत सुन्दर है।", "mujhe kal subah 7 baje ka alarm set karna hai"]
    return BPETokenizer.train(text * 20, vocab_size=300)


@pytest.mark.parametrize("model_type", ["gpt", "lstm"])
def test_streamed_text_equals_full_decode(tok, model_type):
    """Streaming must give exactly the same text as decoding all tokens at once — including when a
    token ends in the middle of a 3-byte Devanagari character."""
    torch.manual_seed(0)
    if model_type == "gpt":
        m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, d_model=32, n_layers=1, n_heads=2))
    else:
        m = LSTMLM(LSTMConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1))
    for seed in range(5):
        pieces = list(stream(m, tok, "भारत", 40, generator=torch.Generator().manual_seed(seed), temperature=1.0))
        ids = [tok.eot_id] + tok.encode_ordinary("भारत")
        out = m.generate(torch.tensor([ids]), 40, generator=torch.Generator().manual_seed(seed),
                         temperature=1.0, eot_id=tok.eot_id)[0].tolist()[len(ids):]
        if out and out[-1] == tok.eot_id:
            out = out[:-1]
        assert "".join(pieces) == tok.decode(out)
        assert complete(m, tok, "भारत", 40, temperature=1.0, top_k=None,
                        generator=torch.Generator().manual_seed(seed)) == "भारत" + tok.decode(out)
