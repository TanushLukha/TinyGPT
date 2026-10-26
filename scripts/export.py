"""Export the two models M6 serves as bundles (safetensors + config.json + tokenizer), then check them.

    python scripts/export.py                  # models/lm      <- checkpoints/gpt-llama/best.pt (M3)
                                              # models/parser  <- runs/m5-annotated/full-long (M5's best)
    python scripts/export.py --verify 1390    # check the parser on the whole validation set
    python scripts/export.py --verify 0       # no checks

The checks run on the CPU, where the service runs:
  1. each bundle gives exactly the logits of the checkpoint it came from;
  2. the parser decoding requests together in one padded batch (as the server does) writes exactly
     what it writes for each request alone (M5's evaluation), on the first --verify validation
     requests — and scores the same exact match.
Also writes models/README.md: the model card shown on the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from finetune import load_best  # noqa: E402

from tinygpt.bpe import BPETokenizer  # noqa: E402
from tinygpt.bundle import load_bundle, save_bundle  # noqa: E402
from tinygpt.lora import merge_lora  # noqa: E402
from tinygpt.models import load_checkpoint  # noqa: E402
from tinygpt.serve.engine import Parser  # noqa: E402
from tinygpt.top import from_model, load, predict, prompt_ids, score  # noqa: E402


def same_logits(a, b, vocab: int) -> float:
    x = torch.randint(0, vocab, (4, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        return (a(x)[0] - b(x)[0]).abs().max().item()


def export_lm(args) -> dict:
    model, ck = load_checkpoint(args.lm)
    results = json.loads((Path(args.lm).parent / "results.json").read_text())
    out = save_bundle(Path(args.out) / "lm", model, args.tokenizer, task={"name": "lm"},
                      source={"checkpoint": args.lm, "run": results["run"], "iter": ck.get("iter"),
                              "tokens_seen": results.get("tokens_seen"), "trained_on": results.get("device"),
                              "minutes": round(results.get("minutes", 0))},
                      scores={"val_bpb": {k: v["bpb"] for k, v in results["val"].items()}})
    if args.verify:
        diff = same_logits(model, load_bundle(out)[0], model.cfg.vocab_size)
        print(f"  check 1: bundle logits = checkpoint logits (max difference {diff})")
        assert diff == 0.0
    return json.loads((out / "config.json").read_text())


def export_parser(args) -> dict:
    run_dir = Path(args.parser)
    cfg = argparse.Namespace(**json.loads((run_dir / "config.json").read_text()))
    results = json.loads((run_dir / "results.json").read_text())
    model = merge_lora(load_best(cfg, run_dir, "cpu")).eval()       # a LoRA run becomes W + ΔW
    out = save_bundle(Path(args.out) / "parser", model, cfg.tokenizer,
                      task={"name": "hinglish-top", "labels": cfg.labels, "target": getattr(cfg, "target", "decoupled")},
                      source={"run": str(run_dir), "method": cfg.method, "base": cfg.base, "steps": cfg.steps,
                              "lr": cfg.lr},
                      scores={"val": results["val"], "test": results.get("test")})
    if args.verify:
        diff = same_logits(model, load_bundle(out)[0], model.cfg.vocab_size)
        print(f"  check 1: bundle logits = checkpoint logits (max difference {diff})")
        assert diff == 0.0
        verify_batching(out, args.verify, args.data_dir)
    return json.loads((out / "config.json").read_text())


def verify_batching(bundle: Path, n: int, data_dir: str) -> None:
    parser = Parser.from_bundle(bundle)
    examples = load("val", data_dir)[:n]
    texts = [e.query for e in examples]
    t0 = time.perf_counter()
    alone = predict(parser.model, parser.tok, [prompt_ids(parser.tok, t) for t in texts], "cpu",
                    max_new_tokens=parser.max_new_tokens)        # M5's path: no padding
    t1 = time.perf_counter()
    together = [r for s in range(0, len(texts), 16) for r in parser.parse(texts[s:s + 16])]
    t2 = time.perf_counter()
    same = sum(from_model(a, parser.labels) == r.written for a, r in zip(alone, together))
    em = score([r.parse for r in together], [e.parse for e in examples])["exact_match"]
    print(f"  check 2: {same}/{len(texts)} validation requests parsed identically in padded batches of 16 "
          f"and alone; exact match {em:.1%} ({t1 - t0:.0f} s alone in same-length groups, {t2 - t1:.0f} s batched)")
    if same != len(texts):
        for a, r in zip(alone, together):
            if from_model(a, parser.labels) != r.written:
                print(f"    differs: {r.text!r}\n      alone:    {from_model(a, parser.labels)}\n      batched:  {r.written}")


def model_card(lm: dict, parser: dict) -> str:
    bpb, test, val = lm["scores"]["val_bpb"], parser["scores"]["test"] or {}, parser["scores"]["val"]
    cfg = lm["model_config"]
    return f"""---
language: [hi, en]
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags: [hindi, hinglish, gpt, from-scratch, semantic-parsing, safetensors]
datasets: [wikimedia/wikipedia, findnitai/english-to-hinglish]
---

# TinyGPT — a Hindi & Hinglish GPT, built from scratch

Two small models from [TinyGPT](https://github.com/TanushLukha/TinyGPT), where every part — the BPE
tokenizer, the Transformer, training, decoding, LoRA and the serving code — is written by hand in PyTorch.

| folder | model | what it does |
|---|---|---|
| `lm/` | `gpt-llama`, {lm['params'] / 1e6:.1f}M parameters | continues Hindi or Hinglish text |
| `parser/` | `gpt-llama-clean` fully fine-tuned on Hinglish-TOP | turns a Hinglish request to a voice assistant into intent + slots |

**Architecture:** decoder-only Transformer, Llama recipe (RoPE, RMSNorm, SwiGLU), {cfg['n_layers']} layers,
d_model {cfg['d_model']}, {cfg['n_heads']} heads, context {cfg['block_size']} tokens, tied embeddings;
8,192-token byte-level BPE with a Devanagari-aware pre-tokeniser.

**Training data:** 18.2M tokens — Hindi Wikipedia (`wikimedia/wikipedia`, 20231101.hi, CC BY-SA) and
Hinglish sentences (`findnitai/english-to-hinglish`, Apache-2.0). The parser: the human-annotated part of
[Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) (Apache-2.0), 2,993
requests; its base model was pretrained without Hinglish-TOP's validation and test requests.

## Scores

| | |
|---|---|
| `lm`: bits per byte, held-out text | {bpb['all']:.3f} (Hindi {bpb['hindi']:.3f}, Hinglish {bpb['hinglish']:.3f}) |
| `parser`: exact match (whole tree right), test, 6,513 requests | {test.get('exact_match', float('nan')):.1%} |
| `parser`: intent accuracy, test | {test.get('intent_acc', float('nan')):.1%} |
| `parser`: slot F1, test | {test.get('slot_f1', float('nan')):.1%} |
| `parser`: exact match, validation | {val['exact_match']:.1%} |

## Use

```python
# pip install torch safetensors "git+https://github.com/TanushLukha/TinyGPT"   (the model's code)
from huggingface_hub import snapshot_download
from tinygpt.serve.engine import TextGenerator, Parser, GenerateParams

path = snapshot_download("<user>/tinygpt")
lm = TextGenerator.from_bundle(f"{{path}}/lm")
print("".join(p.text for p in lm.stream("भारत की राजधानी", GenerateParams(max_new_tokens=60))
              if hasattr(p, "text")))
print(Parser.from_bundle(f"{{path}}/parser").parse(["kal subah 7 baje ka alarm laga do"])[0].parse)
```

Each folder holds `model.safetensors`, `config.json` (architecture, task, provenance, scores, SHA-256 of
each file — checked on load) and `tokenizer.json`.

## Limitations

A {lm['params'] / 1e6:.1f}M-parameter model trained for {lm['source'].get('minutes', '?')} minutes on a laptop: it writes fluent Hindi
but has learned the language, not the facts — names, dates and places in its text are invented
([what it knows](https://github.com/TanushLukha/TinyGPT/blob/main/docs/results/m6_facts.md)) — and greedy
decoding repeats itself. The parser gets fewer than half of
unseen requests exactly right; its most common mistake is a slot missing, extra or mislabelled, and about
one mistake in five puts words in a slot that the request doesn't contain. For learning and demos, not for
products.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default="checkpoints/gpt-llama/best.pt")
    ap.add_argument("--parser", default="runs/m5-annotated/full-long", help="an M5 run directory")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--out", default="models")
    ap.add_argument("--verify", type=int, default=300, help="validation requests for check 2 (0 = no checks)")
    ap.add_argument("--data-dir", default="data/top")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    BPETokenizer.load(args.tokenizer)                       # fail early if it is missing

    print(f"lm <- {args.lm}")
    lm = export_lm(args)
    print(f"parser <- {args.parser}")
    parser = export_parser(args)
    (Path(args.out) / "README.md").write_text(model_card(lm, parser))
    for name in ("lm", "parser"):
        size = sum(f.stat().st_size for f in (Path(args.out) / name).iterdir())
        print(f"-> {args.out}/{name}/  {size / 1e6:.1f} MB")
    print(f"-> {args.out}/README.md (model card)")


if __name__ == "__main__":
    main()
