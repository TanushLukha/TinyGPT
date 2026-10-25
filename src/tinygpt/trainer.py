"""The training loop shared by every neural model in this repo (M2's LSTM, M3's GPT).

    train(model, "lstm", TrainConfig(run="lstm"))  ->  checkpoints/lstm/
        config.json     model + training settings
        metrics.jsonl   one line per log step ("train") and per evaluation ("val")
        best.pt         weights with the lowest validation loss so far
        last.pt         everything needed to resume exactly: weights, optimiser, step, RNG states
        results.json    final scores of best.pt (loss, bits per byte; overall, Hindi, Hinglish)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import TokenData
from .device import pick_device
from .metrics import evaluate


@dataclass
class TrainConfig:
    data_dir: str = "data/tokens"
    out_dir: str = "checkpoints"
    run: str = "lstm"
    device: str = "auto"
    seed: int = 1337
    # each step: batch_size random windows of block_size tokens = 8,192 tokens
    batch_size: int = 32
    block_size: int = 256
    max_iters: int = 4000
    # AdamW; learning rate warms up linearly, then decays along a cosine to min_lr
    lr: float = 1e-2            # measured: beat 3e-3 at every evaluation (docs/M2_baselines.md)
    min_lr: float = 1e-3
    warmup_iters: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    # logging and evaluation (evaluation always uses ALL validation tokens)
    log_interval: int = 50
    eval_interval: int = 250
    eval_batch_size: int = 32


def lr_at(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    progress = min(1.0, (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay only on matrices (embeddings, layer weights), never on biases or norm gains."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def _rng_state(device: torch.device, rng: np.random.Generator) -> dict:
    state = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: dict, device: torch.device, rng: np.random.Generator) -> None:
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _save(obj: dict, path: Path) -> None:
    """Write to a temporary file, then rename: a crash mid-save never leaves a broken checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(model: torch.nn.Module, model_type: str, cfg: TrainConfig, resume: bool = False,
          stop_iter: int | None = None, log=print) -> dict | None:
    """Train `model`; returns the contents of results.json (None if stopped early by stop_iter,
    which exists so tests can simulate an interrupted run)."""
    device = pick_device(cfg.device)
    run_dir = Path(cfg.out_dir) / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data, val_data = TokenData(cfg.data_dir, "train"), TokenData(cfg.data_dir, "val")
    if train_data.vocab_size != model.cfg.vocab_size:
        raise ValueError(f"data has vocab {train_data.vocab_size}, model {model.cfg.vocab_size}")
    if getattr(model.cfg, "block_size", cfg.block_size) < cfg.block_size:
        raise ValueError(f"model handles {model.cfg.block_size} tokens, training windows are {cfg.block_size}")

    model.to(device)
    opt = make_optimizer(model, cfg)
    rng = np.random.default_rng(cfg.seed)          # picks the training windows
    metrics_path = run_dir / "metrics.jsonl"
    start, best, best_eval, elapsed = 0, math.inf, None, 0.0
    if resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        _set_rng_state(ck["rng"], device, rng)
        start, best, best_eval, elapsed = ck["iter"], ck["best_val_loss"], ck["best_eval"], ck["elapsed"]
        kept = [ln for ln in metrics_path.read_text().splitlines() if json.loads(ln)["iter"] <= start]
        metrics_path.write_text("".join(ln + "\n" for ln in kept))
        log(f"resuming {run_dir} from iter {start}")
    else:
        metrics_path.write_text("")
        (run_dir / "config.json").write_text(json.dumps(
            {"model_type": model_type, "model": asdict(model.cfg), "train": asdict(cfg)}, indent=2))

    tokens_per_iter = cfg.batch_size * cfg.block_size
    n_params = model.num_params()
    log(f"{model_type}: {n_params / 1e6:.2f}M parameters ({model.num_params(non_embedding=True) / 1e6:.2f}M "
        f"non-embedding) | device {device.type} | {tokens_per_iter:,} tokens/iter | {cfg.max_iters} iters "
        f"= {cfg.max_iters * tokens_per_iter / len(train_data):.2f} epochs of {len(train_data):,} tokens")

    def checkpoint(it: int) -> dict:
        return {"model_type": model_type, "model_config": asdict(model.cfg), "model": model.state_dict(),
                "train_config": asdict(cfg), "iter": it, "best_val_loss": best, "best_eval": best_eval,
                "elapsed": elapsed}

    mf = open(metrics_path, "a")
    t0 = time.time() - elapsed                    # wall clock, carried across resumes
    t_log, loss_sum, n_sum, gnorm = time.time(), torch.zeros((), device=device), 0, torch.zeros(())
    it = start
    model.train()
    while True:
        if (it % cfg.eval_interval == 0 or it == cfg.max_iters) and not (resume and it == start):
            ev = evaluate(model, val_data, cfg.block_size, device, cfg.eval_batch_size)
            elapsed = time.time() - t0
            is_best = ev["all"]["loss"] < best
            row = {"kind": "val", "iter": it, "tokens": it * tokens_per_iter, "minutes": elapsed / 60,
                   "loss": ev["all"]["loss"], "bpb": ev["all"]["bpb"],
                   **{f"{k}_bpb": v["bpb"] for k, v in ev.items() if k != "all"}}
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            langs = ", ".join(f"{k} {v['bpb']:.3f}" for k, v in ev.items() if k != "all")
            log(f"eval {it:>6} | val loss {ev['all']['loss']:.4f} | {ev['all']['bpb']:.4f} bits/byte "
                f"({langs}){'  *best*' if is_best else ''}")
            if is_best:
                best, best_eval = ev["all"]["loss"], {"iter": it, **ev}
                _save({**checkpoint(it), "optimizer": None, "rng": None}, run_dir / "best.pt")
            _save({**checkpoint(it), "optimizer": opt.state_dict(), "rng": _rng_state(device, rng)},
                  run_dir / "last.pt")
            model.train()
            t_log = time.time()                   # don't count evaluation time as training speed
        if it >= cfg.max_iters or it == stop_iter:
            break

        for group in opt.param_groups:
            group["lr"] = lr_at(it, cfg)
        x, y = train_data.batch(cfg.batch_size, cfg.block_size, rng)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        loss_sum += loss.detach()                 # stays on the GPU: no waiting here
        n_sum += 1
        it += 1

        if it % cfg.log_interval == 0:
            train_loss = (loss_sum / n_sum).item()   # .item() waits for the GPU, so only every log step
            dt = time.time() - t_log
            tok_s = n_sum * tokens_per_iter / max(dt, 1e-9)
            eta = (cfg.max_iters - it) * dt / n_sum / 60
            row = {"kind": "train", "iter": it, "tokens": it * tokens_per_iter, "loss": train_loss,
                   "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm), "tok_per_s": tok_s}
            mf.write(json.dumps(row) + "\n")
            log(f"iter {it:>6}/{cfg.max_iters} | loss {train_loss:.4f} | lr {row['lr']:.2e} | "
                f"grad norm {row['grad_norm']:.2f} | {tok_s / 1e3:.1f}k tok/s | {eta:.1f} min left")
            t_log, loss_sum, n_sum = time.time(), torch.zeros((), device=device), 0
    mf.close()
    if stop_iter is not None and it == stop_iter and it < cfg.max_iters:
        return None

    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])            # leave the best weights in `model`
    results = {"run": cfg.run, "model_type": model_type, "params": n_params,
               "non_embedding_params": model.num_params(non_embedding=True),
               "best_iter": best_eval["iter"], "tokens_seen": best_eval["iter"] * tokens_per_iter,
               "minutes": elapsed / 60, "device": device.type,
               "val": {k: v for k, v in best_eval.items() if k != "iter"},
               "model_config": asdict(model.cfg), "train_config": asdict(cfg)}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"best: iter {results['best_iter']} | {results['val']['all']['bpb']:.4f} bits/byte -> {run_dir}/")
    return results
