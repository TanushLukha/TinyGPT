"""What does the language model know? Twenty capitals (M6).

    python scripts/probe_facts.py            # -> docs/results/m6_facts.md

The demo writes fluent Hindi with invented facts ("भारत की राजधानी जेटन वाजिदा है"). Is that the
serving code or the model? This script asks the model directly. For each place, "<place> की
राजधानी" ("the capital of <place>") is followed by each of the 20 capitals, and the model's
likelihood ranks them: is the right one first? Then what it actually writes: greedily, and in 10
samples with the demo's default settings. Last, how often its training text states the fact
(data/clean/train.jsonl) and how often the city appears at all.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygpt.serve.engine import GenerateParams, TextGenerator

FACTS = [("भारत", "दिल्ली"), ("महाराष्ट्र", "मुंबई"), ("बिहार", "पटना"), ("उत्तर प्रदेश", "लखनऊ"),
         ("राजस्थान", "जयपुर"), ("मध्य प्रदेश", "भोपाल"), ("पश्चिम बंगाल", "कोलकाता"), ("तमिलनाडु", "चेन्नई"),
         ("गुजरात", "गांधीनगर"), ("पंजाब", "चंडीगढ़"), ("ओडिशा", "भुवनेश्वर"), ("असम", "दिसपुर"),
         ("केरल", "तिरुवनंतपुरम"), ("झारखंड", "रांची"), ("नेपाल", "काठमांडू"), ("पाकिस्तान", "इस्लामाबाद"),
         ("जापान", "टोक्यो"), ("फ्रांस", "पेरिस"), ("चीन", "बीजिंग"), ("रूस", "मास्को")]
CITIES = [city for _, city in FACTS]
PLACES = [place for place, _ in FACTS]


@torch.no_grad()
def logprob(lm: TextGenerator, prompt: str, continuation: str) -> float:
    """log P(continuation | <|endoftext|> prompt), summed over its tokens."""
    p = [lm.tok.eot_id] + lm.tok.encode_ordinary(prompt)
    c = lm.tok.encode_ordinary(continuation)
    logits = lm.model(torch.tensor([p + c], device=lm.device))[0][0]
    lp = torch.log_softmax(logits[len(p) - 1:-1].float(), dim=-1)
    return lp[torch.arange(len(c)), torch.tensor(c, device=lp.device)].sum().item()


def write(lm: TextGenerator, prompt: str, params: GenerateParams) -> str:
    return "".join(e.text for e in lm.stream(prompt, params) if hasattr(e, "text")).strip()


def names(text: str, city: str) -> bool:
    return city in " ".join(text.split()[:3])          # the city among the first three words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/lm")
    ap.add_argument("--corpus", default="data/clean/train.jsonl", help="the text the model was trained on")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--out", default="docs/results/m6_facts.md")
    args = ap.parse_args()

    lm = TextGenerator.from_bundle(args.model)
    corpus = Path(args.corpus).read_text(encoding="utf-8") if Path(args.corpus).exists() else None
    # the demo's default decoding, and a more conservative one (M4: min-p keeps the likelier tokens)
    settings = {"demo": dict(temperature=0.8, top_k=50), "min-p": dict(temperature=0.7, top_k=None, min_p=0.1)}
    rows, right, rank_sum, greedy_right, rev_right, rev_sum = [], 0, 0, 0, 0, 0
    sampled_right = dict.fromkeys(settings, 0)
    for place, city in FACTS:
        prompt = f"{place} की राजधानी"
        scores = {c: logprob(lm, prompt, " " + c) for c in CITIES}
        order = sorted(CITIES, key=lambda c: -scores[c])
        rank = order.index(city) + 1
        # the other way round, Wikipedia's usual sentence: "<city> <place> की राजधानी है" — which place?
        rev = {p: logprob(lm, city, f" {p} की राजधानी है") for p in PLACES}
        rev_rank = sorted(PLACES, key=lambda p: -rev[p]).index(place) + 1
        rev_right += rev_rank == 1
        rev_sum += rev_rank
        greedy = write(lm, prompt, GenerateParams(max_new_tokens=8, temperature=0))
        hits = {name: sum(names(write(lm, prompt, GenerateParams(max_new_tokens=8, seed=s, **kw)), city)
                          for s in range(args.samples)) for name, kw in settings.items()}
        right += rank == 1
        rank_sum += rank
        greedy_right += names(greedy, city)
        for name in settings:
            sampled_right[name] += hits[name]
        seen = f"{corpus.count(f'{prompt} {city}')} | {corpus.count(city):,}" if corpus else "– | –"
        rows.append(f"| {prompt} | {city} | {rank} | {order[0]} | {rev_rank} | {' '.join(greedy.split()[:4])} … | "
                    f"{hits['demo']}/{args.samples} | {hits['min-p']}/{args.samples} | {seen} |")
        print(rows[-1])
    n = len(FACTS)
    total = n * args.samples
    summary = (f"The right capital ranks first for **{right} of {n}** places (chance: 1 in {n}); its average "
               f"rank is {rank_sum / n:.1f} of {n} (chance: {(n + 1) / 2:.1f}). The greedy text names it for "
               f"{greedy_right} of {n} places; samples name it {sampled_right['demo']} times in {total} with the "
               f"demo's settings (temperature 0.8, top-k 50) and {sampled_right['min-p']} in {total} with "
               f"temperature 0.7, min-p 0.1. The other way round (`<city> ___ की राजधानी है`: which place?), the "
               f"right place ranks first for {rev_right} of {n}, average rank {rev_sum / n:.1f}.")
    lines = ["# M6 — what the language model knows: 20 capitals", "",
             "`scripts/probe_facts.py`. Prompt: `<place> की राजधानी` (\"the capital of <place>\"). *Rank*: where "
             "the model's likelihood puts the right city among all 20; *pick*: its first choice; *reverse*: the "
             "right place's rank the other way round, after `<city>` in `<city> <place> की राजधानी है`; *greedy*: the "
             f"text it writes at temperature 0; *demo*, *min-p*: how many of {args.samples} samples name the "
             "right city in their first three words, with the demo's default settings (temperature 0.8, top-k 50) "
             "or temperature 0.7 and min-p 0.1; *fact*, *city*: how often the training text contains "
             "`<place> की राजधानी <city>`, and the city at all.", "",
             summary, "",
             "| prompt | right city | rank | pick | reverse | greedy | demo | min-p | fact | city |",
             "|---|---|---:|---|---:|---|---:|---:|---:|---:|", *rows, ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"\n{summary}\n-> {args.out}")


if __name__ == "__main__":
    main()
