"""Is the browser version the same model? The JavaScript port (web/tinygpt.js) against PyTorch,
with the real bundles (needs Node.js).

    python scripts/check_web.py                # -> docs/results/m6_web.md (~7 minutes)
    python scripts/check_web.py --requests 50  # quicker

1. tokenizer: token ids of every line of the validation text and every validation request
2. language model: logits after real prompts (largest difference), and greedy continuations
3. parser: the trees written for validation requests, and their exact match
4. speed: milliseconds per token in Node.js (a browser runs the same JavaScript engine)
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from tinygpt.serve.engine import GenerateParams, Parser, TextGenerator
from tinygpt.top import load, score

NODE = shutil.which("node")
PROMPTS = ["भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje",
           "हिन्दी भाषा", "गंगा नदी", "क्रिकेट", "aaj ka weather", "एक छोटे से गाँव में", "होली का त्योहार"]


def js(command: str, payload: dict) -> object:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    out = subprocess.run([NODE, "web/test/cli.mjs", command, f.name], capture_output=True, text=True, check=True)
    Path(f.name).unlink()
    return json.loads(out.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models")
    ap.add_argument("--requests", type=int, default=200, help="validation requests to parse in both")
    ap.add_argument("--data-dir", default="data/top")
    ap.add_argument("--out", default="docs/results/m6_web.md")
    args = ap.parse_args()
    if NODE is None:
        raise SystemExit("needs Node.js: https://nodejs.org (or `brew install node`)")
    lm_dir, parser_dir = Path(args.models) / "lm", Path(args.models) / "parser"
    lm, parser = TextGenerator.from_bundle(lm_dir), Parser.from_bundle(parser_dir)
    rows = []

    # 1. tokenizer
    texts = [line for p in ("data/clean/val.jsonl",) if Path(p).exists()
             for doc in Path(p).read_text(encoding="utf-8").splitlines() for line in json.loads(doc)["text"].split("\n")]
    val = load("val", args.data_dir)
    texts += [e.query for e in val]
    got = js("tokenize", {"tokenizer": str(lm_dir / "tokenizer.json"), "texts": texts})
    same = sum(g == lm.tok.encode_ordinary(t) for g, t in zip(got, texts))
    n_tokens = sum(len(g) for g in got)
    rows.append(f"| tokenizer: texts encoded to the same token ids | {same:,} of {len(texts):,} ({n_tokens:,} tokens) |")
    print(rows[-1])

    # 2. the language model: logits, then greedy text
    worst = 0.0
    for p in PROMPTS:
        ids = [lm.tok.eot_id] + lm.tok.encode_ordinary(p)
        with torch.no_grad():
            expected = lm.model(torch.tensor([ids]))[0][0]
        worst = max(worst, (torch.tensor(js("logits", {"bundle": str(lm_dir), "ids": ids})) - expected).abs().max().item())
    rows.append(f"| language model: largest logit difference, {len(PROMPTS)} prompts | {worst:.1e} |")
    print(rows[-1])
    same_text = 0
    for p in PROMPTS:
        want = "".join(e.text for e in lm.stream(p, GenerateParams(max_new_tokens=60, temperature=0)) if hasattr(e, "text"))
        got = js("stream", {"bundle": str(lm_dir), "prompt": p, "params": {"maxNewTokens": 60, "temperature": 0}})
        same_text += got["text"] == want
    rows.append(f"| language model: identical 60-token greedy continuations | {same_text} of {len(PROMPTS)} |")
    print(rows[-1])

    # 3. the parser
    examples = val[:args.requests]
    queries = [e.query for e in examples]
    want = [r for s in range(0, len(queries), 16) for r in parser.parse(queries[s:s + 16])]
    got = js("parse", {"bundle": str(parser_dir), "texts": queries})
    same_tree = sum(g["written"] == w.written for g, w in zip(got, want))
    em_js = score([g["parse"] for g in got], [e.parse for e in examples])["exact_match"]
    em_py = score([w.parse for w in want], [e.parse for e in examples])["exact_match"]
    rows.append(f"| parser: identical trees, first {len(queries)} validation requests | {same_tree} of {len(queries)} |")
    rows.append(f"| parser: exact match, browser / server | {em_js:.1%} / {em_py:.1%} |")
    print("\n".join(rows[-2:]))

    # 4. speed
    speed = js("speed", {"bundle": str(lm_dir), "n": 100})
    rows.append(f"| speed in Node.js (one thread): time per token | {speed['msPerToken']:.1f} ms |")
    print(rows[-1])

    node = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip()
    Path(args.out).write_text("\n".join([
        "# M6 — the browser version against PyTorch", "",
        "`scripts/check_web.py`: `web/tinygpt.js` (the model rewritten in JavaScript, which the demo page runs) "
        "against the Python/PyTorch code it was ported from, with the real bundles. "
        f"{platform.system()} {platform.machine()}, Node.js {node}, PyTorch {torch.__version__}.", "",
        "| check | result |", "|---|---|", *rows, ""]))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
