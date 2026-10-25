import math

import pytest
import torch

from tinygpt.models import GPT, GPTConfig
from tinygpt.models.gpt import RMSNorm, apply_rope, rope_tables

STYLES = ["gpt2", "llama"]


def tiny(style="gpt2", **kw) -> GPT:
    cfg = dict(vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0, style=style)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg))


@pytest.mark.parametrize("style", STYLES)
def test_manual_attention_matches_fused(style):
    """The hand-written attention and PyTorch's fused kernel compute the same function."""
    torch.manual_seed(0)
    fused = tiny(style, attn="sdpa")
    manual = tiny(style, attn="manual")
    manual.load_state_dict(fused.state_dict())
    idx = torch.randint(0, 50, (3, 16))
    lf, loss_f = fused(idx, idx)
    lm, loss_m = manual(idx, idx)
    torch.testing.assert_close(lm, lf, atol=1e-5, rtol=1e-4)
    gf = torch.autograd.grad(loss_f, list(fused.parameters()))
    gm = torch.autograd.grad(loss_m, list(manual.parameters()))
    for a, b in zip(gm, gf):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("attn", ["sdpa", "manual"])
def test_causal(style, attn):
    """Changing token t must not change any prediction made before t."""
    torch.manual_seed(0)
    m = tiny(style, attn=attn).eval()
    idx = torch.randint(0, 50, (1, 16))
    changed = idx.clone()
    changed[0, 9] = (idx[0, 9] + 1) % 50
    a, _ = m(idx)
    b, _ = m(changed)
    torch.testing.assert_close(a[:, :9], b[:, :9])
    assert not torch.allclose(a[:, 9:], b[:, 9:])


def test_rope_is_relative():
    """After RoPE, q·k depends only on the distance between positions, not on where they are."""
    torch.manual_seed(0)
    cos, sin = rope_tables(8, 64)
    q, k = torch.randn(8), torch.randn(8)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, 8), cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k.view(1, 1, 1, 8), cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert score(5, 2) == pytest.approx(score(40, 37), rel=1e-5)     # same distance, same score
    assert score(5, 2) != pytest.approx(score(5, 4), rel=1e-3)
    x = torch.randn(1, 1, 10, 8)
    assert torch.allclose(apply_rope(x, cos[:10], sin[:10]).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rmsnorm():
    x = torch.randn(4, 32) * 5 + 3
    y = RMSNorm(32)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


@pytest.mark.parametrize("style", STYLES)
def test_parameter_count(style):
    d, L, V, T = 384, 6, 8192, 256
    m = GPT(GPTConfig(vocab_size=V, block_size=T, d_model=d, n_layers=L, n_heads=6, style=style))
    assert m.head.weight is m.wte.weight
    if style == "gpt2":       # per block: qkv + proj + fc + proj weights and biases, two LayerNorms
        block = (3 * d * d + 3 * d) + (d * d + d) + (4 * d * d + 4 * d) + (4 * d * d + d) + 2 * 2 * d
        assert m.num_params(non_embedding=True) == L * block + 2 * d
        assert m.num_params() == L * block + 2 * d + V * d + T * d
    else:                     # no biases, SwiGLU hidden size 1024 = 8d/3, RMSNorm gains only
        block = 4 * d * d + 3 * d * 1024 + 2 * d
        assert m.num_params(non_embedding=True) == L * block + d
        assert m.num_params() == L * block + d + V * d


@pytest.mark.parametrize("style", STYLES)
def test_initial_loss_is_log_vocab(style):
    torch.manual_seed(0)
    m = GPT(GPTConfig(style=style)).eval()
    idx = torch.randint(0, 8192, (2, 128))
    _, loss = m(idx, torch.randint(0, 8192, (2, 128)))
    assert abs(loss.item() - math.log(8192)) < 0.15


def test_too_long_input_is_an_error():
    m = tiny()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))


@pytest.mark.parametrize("style", STYLES)
def test_generate(style):
    torch.manual_seed(0)
    m = tiny(style)
    prompt = torch.tensor([[1, 2, 3]])
    a = m.generate(prompt, 30, temperature=0)            # longer than block_size: context is cropped
    assert a.shape == (1, 33) and torch.equal(a, m.generate(prompt, 30, temperature=0))
    out = m.generate(prompt, 30, temperature=1.0, eot_id=7, generator=torch.Generator().manual_seed(1))
    assert out.shape[1] == 33 or out[0, -1].item() == 7


@pytest.mark.parametrize("style", STYLES)
def test_overfits_a_repeating_pattern(style):
    torch.manual_seed(0)
    m = tiny(style)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    seq = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2]])
    for _ in range(150):
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_attention_maps_are_recorded():
    torch.manual_seed(0)
    m = tiny().eval()
    for block in m.blocks:
        block.attn.record = True
    m(torch.randint(0, 50, (1, 10)))
    att = m.blocks[0].attn.last_attn
    assert att.shape == (1, 4, 10, 10)
    assert torch.allclose(att.sum(-1), torch.ones(1, 4, 10))
    assert torch.all(att.triu(diagonal=1) == 0)           # nothing attends to the future


ACCEL = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(ACCEL is None, reason="no GPU here")
@pytest.mark.parametrize("style", STYLES)
def test_gpu_matches_cpu(style):
    torch.manual_seed(0)
    m = tiny(style)
    idx = torch.randint(0, 50, (2, 16))
    logits_cpu, loss_cpu = m(idx, idx)
    loss_cpu.backward()
    g_cpu = m.wte.weight.grad.clone()
    m.zero_grad()
    m.to(ACCEL)
    logits, loss = m(idx.to(ACCEL), idx.to(ACCEL))
    loss.backward()
    torch.testing.assert_close(logits.cpu(), logits_cpu, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m.wte.weight.grad.cpu(), g_cpu, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(ACCEL is None, reason="no GPU here")
@pytest.mark.parametrize("style", STYLES)
def test_gpu_training_step_with_dropout(style):
    """One real optimiser step on the GPU with dropout on (attention, residual, embeddings)."""
    torch.manual_seed(0)
    m = tiny(style, dropout=0.1).to(ACCEL).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, 50, (4, 16), device=ACCEL)
    _, loss = m(idx, idx)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item() and all(torch.isfinite(p).all().item() for p in m.parameters())
