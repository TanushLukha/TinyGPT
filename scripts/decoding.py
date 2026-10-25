"""Same model, different decoding strategies: how they trade likelihood for variety.

    python scripts/decoding.py                          # checkpoints/gpt-llama/best.pt

Prompts are the first 8 tokens of 10 validation documents (5 Hindi, 5 Hinglish). For each strategy,
3 continuations of 80 tokens per prompt (1 for the deterministic ones), measured by:
  nll         average −ln p of the chosen tokens under the model (low = text the model finds likely)
  distinct-2  share of token pairs in a continuation that are different (low = repetitive)
  repeat-4    share of 4-token sequences that already occurred earlier in the same continuation
The first row scores what the documents really say next, as the reference a strategy should match.

Then two checks of why truncation (top-k, top-p, min-p) helps:
  the tail   over all validation text, how much probability each rule removes, and how often the
             token that really comes next is among the removed ones
  drift      the model's uncertainty (entropy, nats) at each step of 200 sampled tokens vs real text:
             does it lose its way when it reads its own samples?
Writes docs/results/m4_decoding.md, with one example per strategy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.dataset import HINDI, TokenData
from tinygpt.device import pick_device
from tinygpt.models import load_checkpoint
from tinygpt.sampling import filter_logits

STRATEGIES = {
    "greedy": dict(temperature=0),
    "greedy + repetition penalty 1.3": dict(temperature=0, repetition_penalty=1.3),
    "temperature 0.7": dict(temperature=0.7),
    "temperature 1.0 (pure sampling)": dict(temperature=1.0),
    "temperature 1.3": dict(temperature=1.3),
    "top-k 50": dict(temperature=1.0, top_k=50),
    "top-p 0.9": dict(temperature=1.0, top_p=0.9),
    "min-p 0.1": dict(temperature=1.0, min_p=0.1),
}


def prompts_from_val(val: TokenData, n_each: int = 5, length: int = 8,
                     follow: int = 80) -> list[tuple[str, list[int], list[int]]]:
    """First `length` tokens of the first documents of each language that are long enough, with the
    (up to) `follow` tokens that really come next in the document."""
    out = []
    for lang_sources, lang in ((HINDI, "hi"), (tuple(s for s in val.ranges if s not in HINDI), "hinglish")):
        found = 0
        for src in lang_sources:
            if src not in val.ranges:
                continue
            s, e = val.ranges[src]
            data = val.data[s:e]
            starts = [0] + [int(i) + 1 for i in np.flatnonzero(data == val.eot_id)[:-1]]
            for st in starts:
                if found < n_each and st + length + 20 < len(data) and val.eot_id not in data[st:st + length]:
                    real = data[st + length:st + length + follow].astype(int).tolist()
                    real = real[:real.index(val.eot_id)] if val.eot_id in real else real
                    out.append((lang, [val.eot_id] + data[st:st + length].astype(int).tolist(), real))
                    found += 1
    return out


TAILS = {"top-k 50": dict(top_k=50), "top-p 0.9": dict(top_p=0.9), "min-p 0.1": dict(min_p=0.1)}


@torch.no_grad()
def tail_stats(model, val: TokenData, block: int, device, batch_size: int = 8) -> dict[str, tuple[float, float]]:
    """Rule -> (average probability it removes, share of real next tokens it removes), temperature 1."""
    removed = {k: [0.0, 0] for k in TAILS}
    n = 0
    for x, y in val.windows(block, batch_size=batch_size):
        logits = model(x.to(device))[0].float().cpu().flatten(0, 1)
        y = y.flatten()
        probs = torch.softmax(logits, dim=-1)
        for name, rule in TAILS.items():
            cut = ~filter_logits(logits, 1.0, **rule).isfinite()
            removed[name][0] += (probs * cut).sum().item()
            removed[name][1] += cut.gather(1, y[:, None]).sum().item()
        n += len(y)
    return {k: (mass / n, real / n) for k, (mass, real) in removed.items()}


DRIFT = {"temperature 1.0 (pure sampling)": dict(temperature=1.0), "top-p 0.9": dict(temperature=1.0, top_p=0.9),
         "min-p 0.1": dict(temperature=1.0, min_p=0.1)}
BINS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200)]


@torch.no_grad()
def drift(model, val: TokenData, device, n_docs: int = 48, prompt_len: int = 8, length: int = 200,
          seed: int = 0) -> dict[str, list[float]]:
    """Mean entropy of the next-token distribution in each step bin, on real text and on samples.

    Uses `n_docs` random documents with at least prompt_len + length tokens (only Hindi Wikipedia
    articles are that long); every strategy continues the same prompts.
    """
    prompts, reals = [], []
    for s, e in val.ranges.values():
        data = val.data[s:e]
        for st in [0] + [int(i) + 1 for i in np.flatnonzero(data == val.eot_id)[:-1]]:
            seg = data[st:st + prompt_len + length]
            if len(seg) == prompt_len + length and val.eot_id not in seg:
                prompts.append([val.eot_id] + seg[:prompt_len].astype(int).tolist())
                reals.append(seg[prompt_len:].astype(int).tolist())
    pick = np.random.default_rng(seed).choice(len(prompts), size=min(n_docs, len(prompts)), replace=False)
    prompts, reals = [prompts[i] for i in pick], [reals[i] for i in pick]

    def entropy(seqs: list[list[int]]) -> list[float]:
        logp = torch.log_softmax(model(torch.tensor(seqs, device=device)[:, :-1])[0].float(), -1)
        ent = (-(logp.exp() * logp).sum(-1))[:, prompt_len:].cpu().numpy()    # before each new token
        return [float(ent[:, a:b].mean()) for a, b in BINS]

    out = {"real text": entropy([p + r for p, r in zip(prompts, reals)])}
    for name, kw in DRIFT.items():
        gen = torch.Generator().manual_seed(seed)
        out[name] = entropy(model.generate(torch.tensor(prompts, device=device), length, generator=gen,
                                           use_cache=True, **kw).tolist())
    return out


def ngrams(seq: list[int], n: int) -> list[tuple]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct2(seq: list[int]) -> float:
    g = ngrams(seq, 2)
    return len(set(g)) / len(g) if g else 1.0


def repeat4(seq: list[int]) -> float:
    g, seen, rep = ngrams(seq, 4), set(), 0
    for x in g:
        rep += x in seen
        seen.add(x)
    return rep / len(g) if g else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/gpt-llama/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--data", default="data/tokens")
    ap.add_argument("--new-tokens", type=int, default=80)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default="docs/results/m4_decoding.md")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load_checkpoint(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)
    val = TokenData(args.data, "val")
    prompts = prompts_from_val(val, follow=args.new_tokens)
    ref = []                                              # the same numbers for the real continuations
    with torch.no_grad():
        for _, p, real in prompts:
            if len(real) >= 4:
                x = torch.tensor([p + real], device=device)
                lp = torch.log_softmax(model(x[:, :-1])[0].float(), -1).gather(-1, x[:, 1:, None])[0, :, 0]
                ref.append((-lp[len(p) - 1:].mean().item(), distinct2(real), repeat4(real)))
    real_name = "real text (what the documents say next)"
    rows = [(real_name, *np.mean(ref, axis=0))]
    examples = {real_name: tok.decode(prompts[0][1][1:] + prompts[0][2])}
    print(f"{real_name:<34} nll {rows[0][1]:.2f}  distinct-2 {rows[0][2]:.2f}  repeat-4 {rows[0][3]:.2f}")
    for name, kw in STRATEGIES.items():
        n = 1 if kw["temperature"] == 0 else args.samples
        batch = torch.tensor([p for _, p, _ in prompts for _ in range(n)], device=device)
        gen = torch.Generator().manual_seed(0)
        out = model.generate(batch, args.new_tokens, generator=gen, use_cache=True, **kw)
        with torch.no_grad():                                 # how likely the model finds its own text
            logp = torch.log_softmax(model(out[:, :-1])[0].float(), -1).gather(-1, out[:, 1:, None])[..., 0]
        logp = logp[:, batch.size(1) - 1:].cpu().numpy()
        stats = []
        for i, row in enumerate(out[:, batch.size(1):].cpu().tolist()):
            cut = row.index(tok.eot_id) if tok.eot_id in row else len(row)
            cont = row[:cut]
            if len(cont) >= 4:
                stats.append((-logp[i, :cut].mean(), distinct2(cont), repeat4(cont)))
            if i == 0:
                examples[name] = tok.decode(prompts[0][1][1:] + cont)
        s = np.array(stats)
        rows.append((name, *s.mean(axis=0)))
        print(f"{name:<34} nll {rows[-1][1]:.2f}  distinct-2 {rows[-1][2]:.2f}  repeat-4 {rows[-1][3]:.2f}")

    tails = tail_stats(model, val, ck["model_config"]["block_size"], device)
    for name, (mass, real) in tails.items():
        print(f"{name:<10} removes {mass:6.1%} of the probability and {real:6.1%} of the real next tokens")
    drifts = drift(model, val, device)
    for name, ent in drifts.items():
        print(f"{name:<34} entropy by step: " + " ".join(f"{v:.2f}" for v in ent))

    table = ["| strategy | nll (lower = more likely) | distinct-2 (higher = more varied) | repeat-4 (lower = less looping) |",
             "|---|---:|---:|---:|"] + [f"| {n} | {a:.2f} | {b:.2f} | {c:.2f} |" for n, a, b, c in rows]
    md = [f"# M4 — decoding strategies ({ck['train_config']['run']})", "",
          f"{len(prompts)} prompts (first 8 tokens of validation documents, half Hindi, half Hinglish); "
          f"{args.new_tokens} new tokens; {args.samples} samples per prompt for the random strategies. "
          "The first row scores what the documents really say next — the text a strategy should resemble.", "",
          "\n".join(table), "", "## The tail",
          "", f"Over all {len(val.data) - 1:,} predicted validation tokens, at temperature 1: the "
          "probability each rule removes, and how often the token that really comes next is removed.", "",
          "| rule | probability removed | real next token removed |", "|---|---:|---:|"]
    md += [f"| {k} | {m:.1%} | {r:.1%} |" for k, (m, r) in tails.items()]
    n_docs = 48
    md += ["", "## Drift", "", f"The model's uncertainty (entropy of its next-token distribution, nats) while it "
           f"reads {n_docs} Hindi Wikipedia articles, and while it reads its own continuations of the same "
           "8-token openings:", "",
           "| text | " + " | ".join(f"steps {a + 1}–{b}" for a, b in BINS) + " |",
           "|---|" + "---:|" * len(BINS)]
    md += [f"| {k} | " + " | ".join(f"{v:.2f}" for v in ent) + " |" for k, ent in drifts.items()]
    md += ["", "## One example per strategy (first Hindi prompt)", ""]
    for name, text in examples.items():
        md += [f"**{name}**", "", "> " + text.replace("\n", " ").strip(), ""]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
