"""Fine-tune a pretrained TinyGPT to parse Hinglish assistant requests (M5).

    python scripts/finetune.py --method lora                    # LoRA, rank 8, attention + MLP layers
    python scripts/finetune.py --method full --lr 1e-4          # update every weight
    python scripts/finetune.py --method scratch --lr 1e-3       # same architecture, random weights
    python scripts/finetune.py --method fewshot                 # no training: 3 solved examples in the prompt
    python scripts/finetune.py --run lora --eval-only --test    # score a saved run on the test set

The task is Hinglish-TOP (src/tinygpt/top.py), downloaded to data/top/ on first use. A run takes
--steps optimizer steps, scores the validation set every --eval-every steps (exact match of greedy
parses), keeps the best weights and writes runs/m5/<run>/: config.json, log.jsonl, adapter.pt
(LoRA: only the adapter) or model.pt (a normal checkpoint), results.json, predictions_val.jsonl and,
with --test, predictions_test.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.dataset import TokenData
from tinygpt.device import pick_device, seed_everything
from tinygpt.lora import apply_lora, count_params, load_lora, lora_state_dict
from tinygpt.metrics import evaluate
from tinygpt.models import build_model, load_checkpoint
from tinygpt.top import (example_ids, fewshot_prompt_ids, from_model, load, make_batch, predict, prompt_ids,
                         score, scored_form)
from tinygpt.trainer import lr_at

DEFAULT_LR = {"lora": 1e-3, "full": 1e-4, "scratch": 1e-3}


def get_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["lora", "full", "scratch", "fewshot"], default="lora")
    ap.add_argument("--run", help="name of the run directory (default: the method)")
    ap.add_argument("--base", default="checkpoints/gpt-llama-clean/best.pt", help="pretrained checkpoint")
    g = ap.add_argument_group("LoRA")
    g.add_argument("--rank", type=int, default=8)
    g.add_argument("--alpha", type=float, default=16.0, help="the update is scaled by alpha / rank")
    g.add_argument("--targets", default="attn,mlp", help="attn, mlp or attn,mlp")
    g.add_argument("--lora-dropout", type=float, default=0.0)
    g = ap.add_argument_group("training")
    g.add_argument("--lr", type=float, help=f"peak learning rate (default {DEFAULT_LR})")
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--warmup", type=int, default=30)
    g.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decay ends at lr × this")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--train-size", type=int, default=0, help="use only this many training examples (0 = all)")
    g.add_argument("--labels", choices=["lower", "upper"], default="lower", help="how the model writes labels")
    g.add_argument("--target", choices=["decoupled", "annotated"], default="decoupled",
                   help="what the model writes: the decoupled tree, or the request annotated with brackets")
    g.add_argument("--shots", type=int, default=3, help="solved examples in the prompt (fewshot)")
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("evaluation")
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--test", action="store_true", help="also score the best weights on the test set")
    g.add_argument("--eval-only", action="store_true", help="score a finished run again (with --test: on test)")
    g.add_argument("--eval-batch", type=int, default=128)
    g.add_argument("--max-new-tokens", type=int, default=256, help="never more than the context window allows")
    g.add_argument("--val-size", type=int, default=0, help="score only this many validation examples (quick checks)")
    g = ap.add_argument_group("paths")
    g.add_argument("--out-dir", default="runs/m5")
    g.add_argument("--data-dir", default="data/top")
    g.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    g.add_argument("--lm-data", default="data/tokens", help="general text, to measure forgetting")
    g.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    args.run = args.run or args.method
    args.lr = args.lr if args.lr is not None else DEFAULT_LR.get(args.method)
    return args


def build(args, device) -> torch.nn.Module:
    """The model this run trains: the pretrained base, the base + LoRA, or fresh random weights."""
    model, ck = load_checkpoint(args.base, device)
    seed_everything(args.seed)                    # LoRA's A matrices / the scratch model's weights
    if args.method == "scratch":
        model = build_model(ck["model_type"], **ck["model_config"]).to(device)
    elif args.method == "lora":
        apply_lora(model, args.rank, args.alpha, args.lora_dropout, tuple(args.targets.split(",")))
    return model


def cpu_state_dict(model) -> dict:
    """The weights on the CPU, with the tied embedding / output matrix still one tensor. (Moving each
    entry of a GPU state dict separately makes two copies of it, and the file grows by 12.6 MB.)"""
    return copy.deepcopy(model).cpu().state_dict()


def save_weights(model, args, run_dir: Path) -> Path:
    if args.method == "lora":
        path = run_dir / "adapter.pt"
        torch.save({"lora": lora_state_dict(model)}, path)
    else:
        path = run_dir / "model.pt"               # an ordinary checkpoint: load_checkpoint() reads it
        torch.save({"model_type": "gpt", "model_config": dataclasses.asdict(model.cfg),
                    "model": cpu_state_dict(model)}, path)
    return path


def load_best(args, run_dir: Path, device) -> torch.nn.Module:
    if args.method == "lora":
        model = build(args, device)
        load_lora(model, torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=True)["lora"])
        return model
    if args.method == "fewshot":
        return load_checkpoint(args.base, device)[0]
    return load_checkpoint(run_dir / "model.pt", device)[0]


def demos(args, train) -> list:
    rng = np.random.default_rng(args.seed)
    return [train[i] for i in rng.choice(len(train), size=args.shots, replace=False)]


def run_eval(model, tok, examples, args, device, train=None) -> tuple[dict, list[tuple[str, str]]]:
    """Greedy parses for `examples` and their scores. Returns (scores, [(what the model wrote,
    the decoupled tree that is scored)])."""
    target = getattr(args, "target", "decoupled")
    if args.method == "fewshot":
        shots = demos(args, train)
        prompts = [fewshot_prompt_ids(tok, shots, e.query, args.labels, target) for e in examples]
        raw = predict(model, tok, prompts, device, args.eval_batch, 48, stop="\n")
    else:
        prompts = [prompt_ids(tok, e.query) for e in examples]
        raw = predict(model, tok, prompts, device, args.eval_batch, args.max_new_tokens)
    outputs = [from_model(r, args.labels) for r in raw]
    preds = [scored_form(o) for o in outputs] if target == "annotated" else outputs
    return score(preds, [e.parse for e in examples]), list(zip(outputs, preds))


def write_predictions(path: Path, examples, preds) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e, (out, pred) in zip(examples, preds):
            row = {"query": e.query, "domain": e.domain, "gold": e.parse, "pred": pred}
            if out != pred:
                row["output"] = out                   # the annotated tree the model wrote
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt(m: dict) -> str:
    return (f"exact match {m['exact_match']:.1%} | intent {m['intent_acc']:.1%} | slot F1 {m['slot_f1']:.1%} | "
            f"valid {m['valid']:.1%}")


def batches(lengths: list[int], batch_size: int, rng: np.random.Generator, group: int = 50):
    """Index lists, forever. Each epoch visits every example once in a new random order; within
    chunks of `group` batches, examples are sorted by length, so a batch holds similar lengths
    (padding: 3% of the compute instead of 57% for random batches). The batches are then shuffled."""
    B = min(batch_size, len(lengths))
    while True:
        order = rng.permutation(len(lengths))
        out = []
        for c in range(0, len(order), B * group):
            chunk = sorted(order[c:c + B * group], key=lambda i: lengths[i])
            out += [chunk[i:i + B] for i in range(0, len(chunk) - B + 1, B)]   # drop a short last batch
        for i in rng.permutation(len(out)):
            yield out[i]


def train(model, tok, train_ex, val_ex, args, device, run_dir: Path) -> dict:
    items = [example_ids(tok, e, args.labels, args.target) for e in train_ex]
    fits = [it for it in items if len(it[0]) <= model.cfg.block_size + 1]   # inputs are 1 shorter
    if len(fits) < len(items):
        print(f"skipping {len(items) - len(fits)} training example(s) longer than the "
              f"{model.cfg.block_size}-token context window", flush=True)
    items = fits
    if not items:
        raise ValueError("no training example fits the context window")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = SimpleNamespace(lr=args.lr, min_lr=args.lr * args.min_lr_ratio, warmup_iters=args.warmup,
                            max_iters=args.steps)
    stream = batches([len(ids) for ids, _ in items], args.batch_size, np.random.default_rng(args.seed))
    best = {"exact_match": -1.0}
    log = open(run_dir / "log.jsonl", "w")
    loss_sum, n, train_s, t0, avg = torch.zeros((), device=device), 0, 0.0, time.time(), float("nan")
    for step in range(1, args.steps + 1):
        ts = time.time()
        for group in opt.param_groups:
            group["lr"] = lr_at(step - 1, sched)
        x, y = make_batch([items[i] for i in next(stream)], tok.eot_id)
        model.train()
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        loss_sum += loss.detach()
        n += 1
        if step % 25 == 0:
            avg = (loss_sum / n).item()           # waits for the GPU: only every 25 steps
            train_s += time.time() - ts
            log.write(json.dumps({"step": step, "loss": avg, "lr": opt.param_groups[0]["lr"]}) + "\n")
            loss_sum, n = torch.zeros((), device=device), 0
        else:
            train_s += time.time() - ts
        if step % args.eval_every == 0 or step == args.steps:
            m, preds = run_eval(model, tok, val_ex, args, device)
            log.write(json.dumps({"step": step, "val": m}) + "\n")
            log.flush()
            is_best = m["exact_match"] > best["exact_match"]
            print(f"step {step:>4}/{args.steps} | train loss {avg:.3f} | val {fmt(m)}{'  *best*' if is_best else ''}",
                  flush=True)
            if is_best:
                best = {**m, "step": step}
                save_weights(model, args, run_dir)
                write_predictions(run_dir / "predictions_val.jsonl", val_ex, preds)
    log.close()
    return {"val": best, "best_step": best.pop("step"), "train_seconds": train_s,
            "minutes": (time.time() - t0) / 60, "train_examples_used": len(items)}


def main(argv=None) -> None:
    args = get_args(argv)
    run_dir = Path(args.out_dir) / args.run
    if args.eval_only:                            # the saved run decides everything but what to score
        saved = json.loads((run_dir / "config.json").read_text())
        args = argparse.Namespace(**{**saved, "test": args.test, "eval_only": True, "device": args.device})
    device = pick_device(args.device)
    tok = BPETokenizer.load(args.tokenizer)
    train_ex, val_ex = load("train", args.data_dir), load("val", args.data_dir)
    val_ex = val_ex[:args.val_size] if args.val_size else val_ex
    if args.train_size:
        keep = np.random.default_rng(1234).permutation(len(train_ex))[:args.train_size]   # same subset every run
        train_ex = [train_ex[i] for i in sorted(keep)]

    if args.eval_only:
        results = json.loads((run_dir / "results.json").read_text())
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        seed_everything(args.seed)
        model = build(args, device)
        trainable, total = count_params(model)
        trainable = 0 if args.method == "fewshot" else trainable
        print(f"{args.run}: {args.method} on {Path(args.base).parent.name} | {trainable:,} of {total:,} weights "
              f"train ({trainable / total:.2%}) | {len(train_ex):,} examples | device {device.type}", flush=True)
        results = {"run": args.run, "method": args.method, "base": args.base, "trainable_params": trainable,
                   "total_params": total, "train_size": len(train_ex), "config": vars(args)}
        if args.method == "fewshot":
            t0 = time.time()
            m, preds = run_eval(model, tok, val_ex, args, device, train_ex)
            write_predictions(run_dir / "predictions_val.jsonl", val_ex, preds)
            results.update(val=m, best_step=0, train_seconds=0.0, minutes=(time.time() - t0) / 60)
            print(f"val {fmt(m)}")
        else:
            results.update(train(model, tok, train_ex, val_ex, args, device, run_dir))
            weights = run_dir / ("adapter.pt" if args.method == "lora" else "model.pt")
            results["weights_bytes"] = weights.stat().st_size
            model = load_best(args, run_dir, device)
            lm = evaluate(model, TokenData(args.lm_data, "val"), model.cfg.block_size, device)
            results["lm_bpb"] = {k: v["bpb"] for k, v in lm.items()}
            print(f"best step {results['best_step']} | val {fmt(results['val'])} | general text after fine-tuning: "
                  f"{results['lm_bpb']['all']:.4f} bits/byte")

    if args.test:
        model = load_best(args, run_dir, device)
        test_ex = load("test", args.data_dir)
        m, preds = run_eval(model, tok, test_ex, args, device, train_ex)
        write_predictions(run_dir / "predictions_test.jsonl", test_ex, preds)
        results["test"] = m
        print(f"test {fmt(m)}")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"-> {run_dir}/")


if __name__ == "__main__":
    main()
