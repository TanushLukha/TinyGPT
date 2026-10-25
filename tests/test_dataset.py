import math

import numpy as np
import pytest
import torch

from tinygpt.dataset import TokenData
from tinygpt.metrics import bits_per_byte, evaluate, token_logprobs, total_nll
from tinygpt.models import LSTMConfig, LSTMLM


def test_ranges_and_bytes(token_dir):
    d = TokenData(token_dir(), "val")
    assert d.ranges == {"wiki_hi": (0, 120), "findnitai": (120, 180)}
    assert d.n_bytes() == 540 and d.n_bytes(["findnitai"]) == 180
    for s, e in d.ranges.values():
        assert d.data[e - 1] == d.eot_id                  # every source ends a document


def test_meta_must_match_file(token_dir):
    path = token_dir()
    np.zeros(5, dtype=np.uint16).tofile(path / "val.bin")
    with pytest.raises(ValueError):
        TokenData(path, "val")


def test_batch_targets_are_inputs_shifted_by_one(token_dir):
    d = TokenData(token_dir(), "train")
    x, y = d.batch(8, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    data = torch.from_numpy(d.data.astype(np.int64))
    for row in x:                                         # each row is a real slice of the file
        starts = [i for i in range(len(data) - 16) if torch.equal(data[i:i + 16], row)]
        assert starts


@pytest.mark.parametrize("block", [1, 7, 16, 59, 200])
def test_windows_predict_every_token_once(token_dir, block):
    d = TokenData(token_dir(), "val")
    start, end = d.ranges["findnitai"]
    xs, ys = [], []
    for x, y in d.windows(block, start, end, batch_size=3):
        assert x.shape == y.shape and x.shape[1] <= block
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    data = torch.from_numpy(d.data.astype(np.int64))
    assert torch.equal(torch.cat(ys), data[start + 1:end])        # targets: start+1 .. end-1, in order
    assert torch.equal(torch.cat(xs), data[start:end - 1])        # inputs: the token before each


class Uniform(torch.nn.Module):
    """Same logits for every token -> p = 1/V everywhere."""
    def __init__(self, V):
        super().__init__()
        self.V = V

    def forward(self, idx, targets=None):
        return torch.zeros(*idx.shape, self.V), None


def test_bits_per_byte():
    assert bits_per_byte(math.log(2) * 80, 10) == pytest.approx(8.0)   # 8 bits per byte = no compression


def test_evaluate_uniform_model(token_dir):
    d = TokenData(token_dir(), "val")
    ev = evaluate(Uniform(d.vocab_size), d, block_size=16, device="cpu")
    for name, (n_tok, n_bytes) in {"hindi": (119, 360), "hinglish": (59, 180), "all": (178, 540)}.items():
        assert ev[name]["tokens"] == n_tok
        assert ev[name]["loss"] == pytest.approx(math.log(64), rel=1e-6)
        assert ev[name]["bpb"] == pytest.approx(n_tok * math.log2(64) / n_bytes, rel=1e-6)


def test_token_logprobs_match_total_nll(token_dir):
    """The per-token scores used for model comparisons add up to the loss used everywhere else."""
    d = TokenData(token_dir(), "val")
    torch.manual_seed(0)
    m = LSTMLM(LSTMConfig(vocab_size=64, d_model=8, n_layers=1, dropout=0.0))
    s, e = d.ranges["findnitai"]
    lp = token_logprobs(m, d, 16, "cpu", s, e)
    nats, count = total_nll(m, d, 16, "cpu", s, e)
    assert len(lp) == count == e - s - 1
    assert -lp.sum() == pytest.approx(nats, rel=1e-5)
