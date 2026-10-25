import json
import math

import pytest
import torch

from tinygpt.models import LSTMConfig, LSTMLM, load_checkpoint
from tinygpt.trainer import TrainConfig, lr_at, make_optimizer, train


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_iters=10, max_iters=110)
    assert lr_at(0, cfg) == pytest.approx(1e-4)          # warm-up starts small ...
    assert lr_at(9, cfg) == pytest.approx(1e-3)          # ... reaches the peak ...
    assert lr_at(60, cfg) == pytest.approx(5.5e-4)       # ... halfway down the cosine ...
    assert lr_at(110, cfg) == pytest.approx(1e-4)        # ... ends at min_lr
    after = [lr_at(i, cfg) for i in range(10, 111)]
    assert all(a >= b for a, b in zip(after, after[1:]))


def test_no_weight_decay_on_biases():
    m = LSTMLM(LSTMConfig(vocab_size=20, d_model=8, n_layers=1))
    opt = make_optimizer(m, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and all(p.dim() == 2 for p in decay["params"])
    assert no_decay["weight_decay"] == 0.0 and all(p.dim() == 1 for p in no_decay["params"])
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(m.parameters()))


def tiny(token_dir, tmp_path, **kw):
    data = token_dir(pattern=[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7])
    cfg = TrainConfig(data_dir=str(data), out_dir=str(tmp_path / "ck"), run="t", device="cpu",
                      batch_size=4, block_size=12, max_iters=40, eval_interval=10, log_interval=5,
                      warmup_iters=5, lr=1e-2, min_lr=1e-3, **kw)
    torch.manual_seed(0)
    return LSTMLM(LSTMConfig(vocab_size=64, d_model=24, n_layers=2, dropout=0.1)), cfg


def test_training_run(token_dir, tmp_path):
    model, cfg = tiny(token_dir, tmp_path)
    res = train(model, "lstm", cfg, log=lambda *a: None)
    run = tmp_path / "ck" / "t"
    assert {"config.json", "metrics.jsonl", "best.pt", "last.pt", "results.json"} <= {p.name for p in run.iterdir()}
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    vals = [r for r in rows if r["kind"] == "val"]
    assert [r["iter"] for r in vals] == [0, 10, 20, 30, 40]
    assert vals[0]["loss"] == pytest.approx(math.log(64), abs=0.1)      # starts near uniform
    assert vals[-1]["loss"] < vals[0]["loss"] - 1.5                     # and learns something
    assert res["val"]["all"]["loss"] == min(r["loss"] for r in vals)
    m, ck = load_checkpoint(run / "best.pt")                            # best.pt rebuilds the model
    assert ck["iter"] == res["best_iter"]


def test_resume_is_exact(token_dir, tmp_path):
    """Stopping at step 20 and resuming must give exactly the same model as never stopping."""
    model_a, cfg_a = tiny(token_dir, tmp_path / "a")
    train(model_a, "lstm", cfg_a, log=lambda *a: None)

    model_b, cfg_b = tiny(token_dir, tmp_path / "b")
    assert train(model_b, "lstm", cfg_b, stop_iter=20, log=lambda *a: None) is None
    model_c, _ = tiny(token_dir, tmp_path / "b")         # a fresh process would rebuild the model ...
    torch.manual_seed(123)                                # ... with whatever global RNG state it has
    train(model_c, "lstm", cfg_b, resume=True, log=lambda *a: None)

    for (name, a), (_, c) in zip(model_a.state_dict().items(), model_c.state_dict().items()):
        assert torch.equal(a, c), name
    read = lambda p: [json.loads(x) for x in (p / "ck" / "t" / "metrics.jsonl").read_text().splitlines()]
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("minutes", "tok_per_s")} for r in rows]
    assert strip(read(tmp_path / "a")) == strip(read(tmp_path / "b"))
