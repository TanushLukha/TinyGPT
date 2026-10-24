"""How many tokens does each tokenizer need for the same Hindi / Hinglish text?

    python scripts/compare_tokenizers.py --ours tokenizer/tinygpt-indic-8k.json

Measured on the validation set, separately for Hindi and Hinglish:
  tokens/word  fertility; lower = more text fits in the model's context window
  bytes/token  compression; higher = each token carries more text
  1-token %    share of word occurrences that are a single token
Compared: our BPE; the same BPE trained with GPT-4's regex (ablation); SentencePiece BPE and
Unigram trained on the same sample with the same vocab size; the GPT-2 / GPT-4 / GPT-4o
tokenizers (tiktoken, needs internet once); raw characters and raw UTF-8 bytes.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from tinygpt.bpe import BPETokenizer
from tinygpt.data import HINDI_SOURCES, HINGLISH_SOURCES, read_jsonl

import importlib.util
_spec = importlib.util.spec_from_file_location("train_tokenizer", Path(__file__).with_name("train_tokenizer.py"))
_tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tt)
sample_texts = _tt.sample_texts


class Counter_:
    """Uniform interface: count(text) and word_count(word) -> number of tokens."""
    def __init__(self, name, count, word_count, note=""):
        self.name, self.count, self.word_count, self.note = name, count, word_count, note


def ours(path: Path, name: str) -> Counter_:
    tok = BPETokenizer.load(path)
    return Counter_(name, lambda t: len(tok.encode_ordinary(t)), lambda w: len(tok.encode_ordinary(" " + w)),
                    f"{tok.vocab_size:,} tokens")


def sentencepiece(texts: list[str], vocab: int, kind: str, workdir: Path) -> Counter_ | None:
    try:
        import sentencepiece as spm
    except ImportError:
        return None
    sample = workdir / "spm_sample.txt"
    if not sample.exists():
        with open(sample, "w", encoding="utf-8") as f:
            for t in texts:
                for line in t.split("\n"):
                    if line.strip():
                        f.write(line + "\n")
    prefix = workdir / f"spm_{kind}"
    try:
        spm.SentencePieceTrainer.train(
            input=str(sample), model_prefix=str(prefix), vocab_size=vocab, model_type=kind,
            character_coverage=0.9995, byte_fallback=True, normalization_rule_name="identity",
            input_sentence_size=2_000_000, shuffle_input_sentence=True, num_threads=os.cpu_count() or 4,
            minloglevel=2)
    except RuntimeError as e:
        print(f"  sentencepiece {kind} skipped: {e}")
        return None
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return Counter_(f"SentencePiece {kind.upper()}", lambda t: len(sp.encode(t)),
                    lambda w: len(sp.encode(w)), f"{vocab:,} tokens, same sample")


def tiktoken_encoders() -> list[Counter_]:
    out = []
    try:
        import tiktoken
    except ImportError:
        return out
    for enc_name, label in (("gpt2", "GPT-2"), ("cl100k_base", "GPT-4 (cl100k)"), ("o200k_base", "GPT-4o (o200k)")):
        try:
            enc = tiktoken.get_encoding(enc_name)
        except Exception:
            print(f"  {label}: not available (tiktoken needs internet the first time)")
            continue
        out.append(Counter_(label, lambda t, e=enc: len(e.encode_ordinary(t)),
                            lambda w, e=enc: len(e.encode_ordinary(" " + w)), f"{enc.n_vocab:,} tokens"))
    return out


def measure(c: Counter_, texts: list[str]) -> tuple[float, float, float]:
    n_tok = sum(c.count(t) for t in texts)
    words = Counter(w for t in texts for w in t.split())
    n_words = sum(words.values())
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    single = sum(f for w, f in words.items() if c.word_count(w) == 1)
    return n_tok / n_words, n_bytes / n_tok, single / n_words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="tokenizer/tinygpt-indic-8k.json")
    ap.add_argument("--train", default="data/clean/train.jsonl")
    ap.add_argument("--val", default="data/clean/val.jsonl")
    ap.add_argument("--sample-mb", type=float, default=40, help="training sample for the ablation / SentencePiece")
    ap.add_argument("--max-eval-mb", type=float, default=3, help="per language, from the validation set")
    ap.add_argument("--no-retrain", action="store_true", help="skip the GPT-4-regex ablation and SentencePiece")
    ap.add_argument("--out", default="docs/results/tokenizers.md")
    args = ap.parse_args()

    ours_path = Path(args.ours)
    tok = BPETokenizer.load(ours_path)
    vocab = tok.vocab_size

    def val_texts(sources):
        out, size = [], 0
        for t, _ in read_jsonl(args.val, sources):
            out.append(t)
            size += len(t.encode("utf-8"))
            if size > args.max_eval_mb * 1e6:
                break
        return out
    evals = {"Hindi": val_texts(HINDI_SOURCES), "Hinglish": val_texts(HINGLISH_SOURCES)}

    counters = [ours(ours_path, "TinyGPT BPE (ours)")]
    if not args.no_retrain:
        texts = sample_texts(Path(args.train), args.sample_mb)
        ablation = ours_path.with_name(ours_path.name.replace("-indic-", "-gpt4-"))
        if not ablation.exists() or ablation == ours_path:
            print("training the GPT-4-regex ablation ...")
            t0 = time.time()
            BPETokenizer.train(texts, vocab, pattern="gpt4").save(ablation)
            print(f"  done in {time.time() - t0:.0f}s")
        counters.append(ours(ablation, "same BPE, GPT-4 regex"))
        with tempfile.TemporaryDirectory() as d:
            for kind in ("bpe", "unigram"):
                print(f"training SentencePiece {kind} ...")
                c = sentencepiece(texts, vocab, kind, Path(d))
                if c:
                    counters.append(c)
    counters += tiktoken_encoders()
    counters.append(Counter_("characters", len, len, "Unicode code points"))
    counters.append(Counter_("UTF-8 bytes", lambda t: len(t.encode("utf-8")),
                             lambda w: len(w.encode("utf-8")), "Devanagari = 3 bytes/char"))

    lines = ["| Tokenizer | Vocab | " + " | ".join(f"{lang} tok/word | {lang} bytes/tok | {lang} 1-tok %"
                                                 for lang in evals) + " |",
             "|---|---|" + "---:|" * (3 * len(evals))]
    for c in counters:
        cells = []
        for lang, texts in evals.items():
            if not texts:
                cells += ["–", "–", "–"]
                continue
            f, b, s = measure(c, texts)
            cells += [f"{f:.2f}", f"{b:.2f}", f"{s:.0%}"]
        lines.append(f"| {c.name} | {c.note} | " + " | ".join(cells) + " |")
    sizes = ", ".join(f"{lang}: {sum(len(t.encode('utf-8')) for t in t_) / 1e6:.2f} MB"
                      for lang, t_ in evals.items())
    table = "\n".join(lines)
    print("\n" + table)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Tokenizer comparison\n\nValidation text — {sizes}. Lower tokens/word is better.\n\n"
                   f"{table}\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
