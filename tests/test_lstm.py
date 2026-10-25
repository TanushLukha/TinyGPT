import math

import pytest
import torch
import torch.nn as nn

from tinygpt.models import LSTMConfig, LSTMLM, ScratchLSTM


def test_scratch_lstm_matches_pytorch():
    torch.manual_seed(0)
    ref = nn.LSTM(5, 7, num_layers=2, batch_first=True)
    ours = ScratchLSTM(5, 7, num_layers=2)
    ours.load_state_dict(ref.state_dict())                # same parameter names and shapes
    x = torch.randn(3, 11, 5, requires_grad=True)
    state = (torch.randn(2, 3, 7), torch.randn(2, 3, 7))
    out_ref, (h_ref, c_ref) = ref(x, state)
    out, (h, c) = ours(x, state)
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(h, h_ref)
    torch.testing.assert_close(c, c_ref)
    g_ref = torch.autograd.grad(out_ref.square().sum(), [x, *ref.parameters()])
    g = torch.autograd.grad(out.square().sum(), [x, *ours.parameters()])
    for a, b in zip(g, g_ref):
        torch.testing.assert_close(a, b)


def test_scratch_and_fused_models_agree():
    torch.manual_seed(0)
    fused = LSTMLM(LSTMConfig(vocab_size=50, d_model=16, n_layers=2, dropout=0.0))
    scratch = LSTMLM(LSTMConfig(vocab_size=50, d_model=16, n_layers=2, dropout=0.0, scratch=True))
    scratch.load_state_dict(fused.state_dict())
    idx = torch.randint(0, 50, (2, 9))
    torch.testing.assert_close(scratch(idx)[0], fused(idx)[0])


def test_parameter_count_and_tying():
    cfg = LSTMConfig(vocab_size=8192, d_model=384, n_layers=2)
    m = LSTMLM(cfg)
    assert m.head.weight is m.emb.weight
    d = cfg.d_model
    per_layer = 4 * d * d + 4 * d * d + 2 * 4 * d       # W_ih, W_hh, two bias vectors
    assert m.num_params(non_embedding=True) == 2 * per_layer
    assert m.num_params() == 8192 * d + 2 * per_layer


def test_forget_gate_bias():
    m = LSTMLM(LSTMConfig(vocab_size=50, d_model=8, n_layers=2, forget_bias=1.0))
    for layer in range(2):
        b = getattr(m.rnn, f"bias_ih_l{layer}") + getattr(m.rnn, f"bias_hh_l{layer}")
        assert torch.allclose(b[8:16], torch.ones(8))


def test_initial_loss_is_log_vocab():
    """A fresh model should be nearly uniform: loss ~ ln V. Much higher means a bad init."""
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig()).eval()
    idx = torch.randint(0, 8192, (4, 64))
    _, loss = m(idx, torch.randint(0, 8192, (4, 64)))
    assert abs(loss.item() - math.log(8192)) < 0.05


def test_generate_carries_state():
    """Feeding tokens one at a time with the state carried = one pass over the whole sequence."""
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=30, d_model=12, n_layers=2, dropout=0.0)).eval()
    idx = torch.randint(0, 30, (1, 10))
    full, _ = m.step(idx)
    state, steps = None, []
    for t in range(10):
        logits, state = m.step(idx[:, t:t + 1], state)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full)


def test_generate_greedy_and_eot_stop():
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=30, d_model=12, n_layers=1, dropout=0.0))
    prompt = torch.tensor([[1, 2, 3]])
    a = m.generate(prompt, 20, temperature=0)
    b = m.generate(prompt, 20, temperature=0)
    assert a.shape == (1, 23) and torch.equal(a, b) and torch.equal(a[:, :3], prompt)
    step = m.step

    def eot_heavy(idx, state=None):                       # make token 29 (our "eot") very likely
        logits, state = step(idx, state)
        logits[..., 29] += 5.0
        return logits, state

    m.step = eot_heavy
    out = m.generate(prompt, 20, temperature=1.0, eot_id=29, generator=torch.Generator().manual_seed(0))
    assert out.shape[1] < 23 and out[0, -1].item() == 29 and 29 not in out[0, 3:-1].tolist()


def test_overfits_a_repeating_pattern():
    """Sanity check of the whole model + optimiser: it must memorise a tiny sequence."""
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=12, d_model=32, n_layers=1, dropout=0.0))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    seq = torch.tensor([[0, 5, 3, 3, 7, 1, 9, 2, 5, 3, 3, 7, 1, 9, 2, 5, 3, 3, 7]])
    for _ in range(150):
        _, loss = m(seq[:, :-1], seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


ACCEL = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(ACCEL is None, reason="no GPU here")
def test_gpu_matches_cpu():
    """nn.LSTM on the GPU (Metal/MPS on a Mac) must agree with the CPU before we trust it for training."""
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=100, d_model=32, n_layers=2, dropout=0.0))
    idx = torch.randint(0, 100, (4, 50))
    logits_cpu, loss_cpu = m(idx, idx)
    loss_cpu.backward()
    g_cpu = m.emb.weight.grad.clone()
    m.zero_grad()
    m.to(ACCEL)
    logits, loss = m(idx.to(ACCEL), idx.to(ACCEL))
    loss.backward()
    torch.testing.assert_close(logits.cpu(), logits_cpu, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m.emb.weight.grad.cpu(), g_cpu, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(ACCEL is None, reason="no GPU here")
def test_gpu_training_step_with_dropout():
    """One real optimiser step on the GPU, with dropout between layers (the training setting)."""
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=100, d_model=32, n_layers=2, dropout=0.1)).to(ACCEL).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, 100, (4, 50), device=ACCEL)
    _, loss = m(idx, idx)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item() and all(torch.isfinite(p).all().item() for p in m.parameters())
