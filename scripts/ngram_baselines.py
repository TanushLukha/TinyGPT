"""Count-based baselines, scored on exactly the same validation predictions as the neural models.

    python scripts/ngram_baselines.py        # about a minute; writes docs/results/m2_ngram.{md,json}

Models: uniform (every token 1/V), then unigram, bigram, trigram and 4-gram with add-k or
Jelinek-Mercer smoothing. The smoothing settings are picked on the validation set; that can only
flatter these baselines, so the neural models' improvement over them is a conservative estimate.

For reference it also runs gzip, bzip2 and xz on the validation text: predicting text and
compressing it are the same problem (a model with X bits/byte could compress text to X/8 of its
size with arithmetic coding).
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tinygpt.bpe import BPETokenizer
from tinygpt.dataset import HINDI, HINGLISH, TokenData
from tinygpt.metrics import summarize
from tinygpt.models.ngram import NGramLM, Smoothing

NAMES = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4-gram"}
K_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
LAMBDA_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05)) + (0.97, 0.99, 0.995, 0.999)
PROMPTS = ["भारत की राजधानी", "yaar kal movie dekhne"]


def score(lm: NGramLM, stats: dict, val: TokenData, sm: Smoothing) -> dict:
    per_src = {src: (float(-np.log(lm.probs(st, sm)).sum()), len(st[0][0])) for src, st in stats.items()}
    return summarize(per_src, val)


COMPRESSORS = {"gzip -9": lambda b: zlib.compress(b, 9), "bzip2 -9": lambda b: bz2.compress(b, 9),
               "xz -9e": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)}


def compressor_rows(val: TokenData, tok: BPETokenizer) -> list[dict]:
    """bits/byte of general-purpose compressors on the validation text (decoded from the tokens)."""
    text = {}
    for src, (s, e) in val.ranges.items():
        text[src] = tok.decode_bytes(t for t in val.data[s:e].tolist() if t != val.eot_id)
        assert len(text[src]) == val.sources[src]["bytes"], "decoded text should match meta.json"
    groups = {"hindi": [s for s in text if s in HINDI], "hinglish": [s for s in text if s in HINGLISH],
              "all": list(text)}
    rows = []
    for name, compress in COMPRESSORS.items():
        bpb = {}
        for g, srcs in groups.items():
            if srcs:
                raw = b"".join(text[s] for s in srcs)
                bpb[g] = 8 * len(compress(raw)) / len(raw)
        rows.append({"model": name, "bpb": bpb})
    return rows


def best_of(lm, stats, val, candidates):
    scored = [(sm, score(lm, stats, val, sm)) for sm in candidates]
    return min(scored, key=lambda pair: pair[1]["all"]["loss"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, val = TokenData(args.data, "train"), TokenData(args.data, "val")
    V = train.vocab_size
    t0 = time.time()
    lm = NGramLM(args.max_order, V, train.eot_id).fit(np.asarray(train.data))
    print(f"counted 1..{args.max_order}-grams of {len(train):,} training tokens in {time.time() - t0:.1f}s: "
          + ", ".join(f"{len(t):,}" for t in lm.tables))
    stats = {src: lm.stats(val.data, s, e) for src, (s, e) in val.ranges.items()}

    rows = []
    uniform = summarize({src: (len(st[0][0]) * math.log(V), len(st[0][0])) for src, st in stats.items()}, val)
    rows.append({"model": "uniform", "smoothing": "1/V for every token", "stored": 0, "val": uniform})
    chosen = {}
    for order in range(1, min(3, args.max_order) + 1):      # add-k, to show where it breaks down
        sm, res = best_of(lm, stats, val, [Smoothing(order, "addk", k=k) for k in K_GRID])
        rows.append({"model": NAMES[order], "smoothing": sm.describe(), "params": asdict(sm),
                     "stored": lm.n_stored(order), "val": res})
        chosen[f"addk{order}"] = sm
    lambdas: tuple = ()
    for order in range(1, args.max_order + 1):           # tune one order at a time, keeping lower ones
        sm, res = best_of(lm, stats, val, [Smoothing(order, "jm", lambdas=lambdas + (lam,)) for lam in LAMBDA_GRID])
        lambdas = sm.lambdas
        rows.append({"model": NAMES.get(order, f"{order}-gram"), "smoothing": sm.describe(), "params": asdict(sm),
                     "stored": lm.n_stored(order), "val": res})
        chosen[f"jm{order}"] = sm

    head = ("| model | smoothing | n-grams stored | loss/token | perplexity | bits/byte | Hindi | Hinglish |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|")
    lines = [head]
    for r in rows:
        v = r["val"]
        lines.append(f"| {r['model']} | {r['smoothing']} | {r['stored']:,} | {v['all']['loss']:.3f} | "
                     f"{math.exp(v['all']['loss']):,.0f} | {v['all']['bpb']:.3f} | {v['hindi']['bpb']:.3f} | "
                     f"{v['hinglish']['bpb']:.3f} |")
    table = "\n".join(lines)
    print("\n" + table)

    samples, compressors = [], []
    tok_path = json.loads((Path(args.data) / "meta.json").read_text())["tokenizer"]
    if Path(tok_path).exists():
        tok = BPETokenizer.load(tok_path)
        compressors = compressor_rows(val, tok)
        ctable = "\n".join(["| compressor | bits/byte | Hindi | Hinglish |", "|---|---:|---:|---:|"] +
                           [f"| {c['model']} | {c['bpb']['all']:.3f} | {c['bpb'].get('hindi', math.nan):.3f} | "
                            f"{c['bpb'].get('hinglish', math.nan):.3f} |" for c in compressors])
        print("\n" + ctable)
        rng = np.random.default_rng(args.seed)
        for key in ("addk2", f"jm{args.max_order}"):
            for prompt in PROMPTS:
                ids = lm.generate([tok.eot_id] + tok.encode_ordinary(prompt), 60, chosen[key], rng)[1:]
                text = tok.decode([t for t in ids if t != tok.eot_id])
                samples.append((f"{NAMES[chosen[key].order]}, {chosen[key].describe()}", prompt, text))
        print()
        for name, prompt, text in samples:
            print(f"--- [{name}] {prompt}\n{text}\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# M2 — n-gram baselines", "",
          f"Counts from {len(train):,} training tokens; scored on all {uniform['all']['tokens']:,} predicted "
          f"validation tokens ({val.n_bytes():,} bytes of text). Lower is better everywhere. Smoothing "
          f"settings were picked on validation (which only flatters these baselines).", "", table, ""]
    if compressors:
        md += ["## For reference: general-purpose compressors", "",
               "Prediction and compression are the same problem: with arithmetic coding, a model that "
               "assigns X bits/byte compresses the text to X/8 of its size. The compressors below only "
               "see the validation text itself; the models above learned from the training set.", "",
               ctable, ""]
    if samples:
        md += ["## Samples (60 tokens, sampled from the model's distribution)", ""]
        for name, prompt, text in samples:
            md += [f"**{name}** — prompt `{prompt}`", "", "> " + text.replace("\n", " ").strip(), ""]
    (out / "m2_ngram.md").write_text("\n".join(md))
    (out / "m2_ngram.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (out / "m2_compressors.json").write_text(json.dumps(compressors, indent=2))
    print(f"-> {out / 'm2_ngram.md'}, {out / 'm2_ngram.json'}")


if __name__ == "__main__":
    main()
