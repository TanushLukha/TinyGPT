"""End-to-end check of scripts/finetune.py on a tiny model, a tiny tokenizer and a tiny dataset."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.models import GPT, GPTConfig, load_checkpoint
from tinygpt.top import example_ids, load

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import finetune  # noqa: E402

ROWS = [("do ghante ke liye alarm set kardo", "[IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] alarm set kardo ]",
         "alarm"),
        ("kal ka weather kaisa hoga", "[IN:GET_WEATHER [SL:DATE_TIME kal ] ka weather kaisa hoga ]", "weather"),
        ("timer ko roko", "[IN:PAUSE_TIMER [SL:METHOD_TIMER timer ] ko roko ]", "timer"),
        ("mujhe Delhi ka weather batao", "[IN:GET_WEATHER mujhe [SL:LOCATION Delhi ] ka weather batao ]", "weather")]


def test_batches_cover_every_example_once_per_epoch_with_little_padding():
    lengths = list(np.random.default_rng(0).integers(10, 200, 1000))
    it = finetune.batches(lengths, 32, np.random.default_rng(0))
    epoch = [i for _ in range(31) for i in next(it)]          # 31 full batches of 32 per epoch
    assert len(epoch) == len(set(epoch)) == 992
    waste = [1 - np.mean([lengths[i] for i in b]) / max(lengths[i] for i in b) for b in (next(it) for _ in range(31))]
    assert np.mean(waste) < 0.1


@pytest.fixture
def setup(tmp_path, token_dir):
    tok = BPETokenizer.train([f"{q} => {p}" for q, p, _ in ROWS] * 30, vocab_size=300)
    tok.save(tmp_path / "tok.json")
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=96, d_model=32, n_layers=2, n_heads=2, style="llama")
    torch.save({"model_type": "gpt", "model_config": dataclasses.asdict(cfg), "model": GPT(cfg).state_dict()},
               tmp_path / "base.pt")
    top = tmp_path / "top"
    top.mkdir()
    header = "en_query\tcs_query\ten_parse\tcs_parse\tdomain\n"
    for split in ("train", "val", "test"):
        (top / f"{split}.tsv").write_text(header + "".join(f"x\t{q}\tx\t{p}\t{d}\n" for q, p, d in ROWS),
                                          encoding="utf-8")
    lm = token_dir(V=tok.vocab_size)
    flags = ["--base", str(tmp_path / "base.pt"), "--tokenizer", str(tmp_path / "tok.json"), "--data-dir", str(top),
             "--lm-data", str(lm), "--out-dir", str(tmp_path / "runs"), "--device", "cpu", "--max-new-tokens", "24"]
    return tmp_path, flags


@pytest.mark.parametrize("method", ["lora", "full", "scratch", "fewshot"])
def test_finetune_runs_and_scores(setup, method):
    tmp_path, flags = setup
    finetune.main(flags + ["--method", method, "--steps", "6", "--eval-every", "3", "--batch-size", "2",
                           "--warmup", "2", "--shots", "2"])
    run = tmp_path / "runs" / method
    res = json.loads((run / "results.json").read_text())
    assert 0.0 <= res["val"]["exact_match"] <= 1.0
    assert len((run / "predictions_val.jsonl").read_text().splitlines()) == len(ROWS)
    if method == "lora":
        assert 0 < res["trainable_params"] < res["total_params"] / 2      # exact counts: test_lora.py
        assert (run / "adapter.pt").exists() and res["weights_bytes"] < 200_000
    if method == "fewshot":
        assert res["trainable_params"] == 0
    if method in ("full", "scratch"):
        model, _ = load_checkpoint(run / "model.pt")         # an ordinary checkpoint
        assert isinstance(model, GPT)
    if method != "fewshot":
        assert res["lm_bpb"]["all"] > 0
    finetune.main(flags + ["--run", method, "--eval-only", "--test"])
    res = json.loads((run / "results.json").read_text())
    assert "test" in res and (run / "predictions_test.jsonl").exists()


def test_annotated_target_and_too_long_examples(setup):
    tmp_path, flags = setup
    finetune.main(flags + ["--method", "lora", "--run", "ann", "--target", "annotated", "--steps", "4",
                           "--eval-every", "2", "--batch-size", "2", "--warmup", "1"])
    res = json.loads((tmp_path / "runs" / "ann" / "results.json").read_text())
    assert res["config"]["target"] == "annotated" and res["train_examples_used"] == len(ROWS)
    row = json.loads((tmp_path / "runs" / "ann" / "predictions_val.jsonl").read_text().splitlines()[0])
    assert row["gold"].startswith("[IN:")                     # scored on the decoupled tree

    # a base model whose context window is too short for the longest example: that example is skipped
    tok = BPETokenizer.load(tmp_path / "tok.json")
    lengths = sorted(len(example_ids(tok, ex, "upper", "annotated")[0]) for ex in load("train", tmp_path / "top"))
    block = lengths[-1] - 2
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block, d_model=32, n_layers=2, n_heads=2, style="llama")
    torch.save({"model_type": "gpt", "model_config": dataclasses.asdict(cfg), "model": GPT(cfg).state_dict()},
               tmp_path / "short.pt")
    finetune.main(flags + ["--base", str(tmp_path / "short.pt"), "--method", "lora", "--run", "short",
                           "--labels", "upper", "--target", "annotated", "--steps", "2", "--eval-every", "2",
                           "--batch-size", "2", "--warmup", "1"])
    res = json.loads((tmp_path / "runs" / "short" / "results.json").read_text())
    assert res["train_examples_used"] == sum(n <= block + 1 for n in lengths) < len(ROWS)


ACCEL = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(ACCEL is None, reason="no GPU here")
def test_saved_model_keeps_tied_weights_once(tmp_path):
    """A model trained on the GPU is saved with its embedding / output matrix stored once."""
    cfg = GPTConfig(vocab_size=4000, block_size=32, d_model=64, n_layers=1, n_heads=2, style="llama")
    model = GPT(cfg).to(ACCEL)
    args = type("args", (), {"method": "full"})
    path = finetune.save_weights(model, args, tmp_path)
    assert path.stat().st_size < 4 * model.num_params() + 4 * 4000 * 64 // 2      # well under two copies
    loaded, _ = load_checkpoint(path)
    assert loaded.head.weight is loaded.wte.weight
