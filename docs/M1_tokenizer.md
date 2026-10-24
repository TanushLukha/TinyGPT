# M1 — Data & tokenizer

**Goal:** turn raw Hindi + Hinglish text into token ids a model can train on, with a tokenizer
written from scratch that is good at Devanagari.

```
Hindi Wikipedia ┐                                   ┌─ tokenizer/tinygpt-indic-8k.json
Hinglish data   ┼─ clean ─ data/clean/*.jsonl ─ BPE ┤
(dev: UD Hindi) ┘   NFC, filter, dedup, split       └─ encode ─ data/tokens/{train,val}.bin
```

## 1. Cleaning the corpus — `text.py`, `data.py`

| Step | Why |
|---|---|
| **NFC normalisation** | The same visible word can be stored as different code-point sequences (e.g. a letter + nukta). Without NFC the tokenizer learns two spellings of one word. |
| Keep **ZWJ / ZWNJ** (U+200D / U+200C), drop other invisible characters | The joiners change how Hindi conjuncts are drawn, so they are part of the text; BOMs, soft hyphens and zero-width spaces are noise. |
| Cut Wikipedia tails (सन्दर्भ, बाहरी कड़ियाँ, …) | Reference lists and link lists are not prose. |
| Devanagari share ≥ 60% (Hindi), Latin share ≥ 80% (Hinglish) | Drops English-heavy pages and mislabelled lines. Counts **letters and combining marks** — a vowel sign is a mark, not a letter. |
| Exact de-duplication | Repeated text is memorised instead of learned, and leaks between train and val. |
| Hash-based train/val split | The same document always goes to the same split, on any machine. |

## 2. Byte-level BPE — `bpe.py`

1. **Pre-tokenise** with a regex into words, numbers, punctuation and spaces. Merges never cross
   these boundaries, so tokens stay meaningful ("the dog" never becomes one token).
2. **Bytes as the base vocabulary** (256 values): any string can be encoded, nothing is unknown.
3. **Train:** repeatedly merge the most frequent adjacent pair into a new token (id 256 + r).
4. **Encode:** merge the pair learned earliest (lowest rank) first, until no learned pair is left —
   this replays exactly what training did.

**The Devanagari bug.** GPT-2/GPT-4 match words with `\p{L}+` (letters). Hindi vowel signs and the
virama are Unicode *combining marks* (`\p{M}`), so those regexes cut words at every vowel sign:

```
GPT-4 regex : 'ह' 'िन' '्द' 'ी' ' भ' 'ाष' 'ा'      ← "हिन्दी भाषा"
TinyGPT     : 'हिन्दी' ' भाषा'
```

Because merges can't cross chunk boundaries, no token can ever cover a whole Hindi word. TinyGPT's
"indic" pattern matches `[\p{L}\p{M}\u200c\u200d]+` instead. Same algorithm, same data, same
vocabulary: **38% fewer tokens per Hindi word** on the full corpus (48% on the small dev corpus).

**Making training fast.** The naive loop recounts every pair in the corpus after every merge:
O(merges × corpus). The trainer here
- counts each *distinct* chunk once, with its frequency (Hindi text repeats words a lot),
- keeps an index pair → chunks that contain it, and only updates those chunks after a merge,
- keeps pair counts in a heap; entries made stale by updates are skipped when popped.

A slow reference trainer in `tests/test_bpe.py` checks that both produce identical merges on 25
random corpora.

**Special tokens** (`<|endoftext|>`, `<|pad|>`, 6 reserved) come after the merges. Reserving them now
means fine-tuning in M5 can add chat or task markers without changing the vocabulary size.

## 3. Measuring a tokenizer — `compare_tokenizers.py`

- **tokens/word (fertility):** lower means more text fits in the context window and each word
  costs less compute.
- **bytes/token:** how much text each token carries.
- **1-token words:** share of word occurrences that are a single token.

The vocabulary is a budget: every merge spent on one script is a merge not spent on another.
That is why the GPT-4-regex tokenizer looks *better* on Hinglish — it can't use merges on Hindi
words, so it spends them on Latin text.

**Full-corpus results** (8,192 tokens; validation text, lower tokens/word is better):

| Tokenizer | Hindi tok/word | Hinglish tok/word | Reading |
|---|---:|---:|---|
| TinyGPT BPE | **1.65** | 1.47 | ours |
| same BPE, GPT-4 regex | 2.67 | 1.34 | the regex alone costs Hindi 62% more tokens |
| SentencePiece BPE / Unigram | 1.67 / 1.64 | 1.52 / 1.53 | ours is at library quality |
| GPT-2 (50k vocab) | 8.11 | 1.69 | built for English: 4.9× our Hindi tokens |
| GPT-4 cl100k (100k vocab) | 5.28 | 1.58 | 3.2× our Hindi tokens |
| GPT-4o o200k (200k vocab) | 1.83 | 1.46 | multilingual; we beat it on Hindi with 1/24 of the vocabulary |

Why the full-corpus numbers are higher than the dev numbers (1.65 vs 1.27): Wikipedia validation
text is far more varied (names, places, technical terms) than the dev news sentences, and the
vocabulary is now shared with twice as much Hinglish.

## 4. Decisions we made, and why

Each entry: what we chose → why → the alternatives and why we didn't pick them.

**1. Corpus: Hindi Wikipedia + two Hinglish datasets.**
Why: Wikipedia is clean, encyclopaedic Hindi under a free licence (CC BY-SA) and can be streamed
from Hugging Face; the Hinglish sets (Apache-2.0) cover how people actually type, which M5 needs.
Not chosen: *IndicCorp / Sangraha* — gigabytes of web text, far more than a small model on a laptop
can use, and noisier; *news crawls* — unclear licences; *Hindi only* — loses the Hinglish angle
that makes the project different.

**2. Hindi capped at 120 MB (≈ 12M tokens).**
Why: enough for a 10–20M-parameter model trained for a few epochs, while one training run stays
around an hour on a fanless MacBook Air; it also keeps Hinglish at roughly a tenth of the data.
Not chosen: *all of Wikipedia* — longer downloads and runs for little gain at this model size
(raise `--max-hindi-mb` later if M3 shows the model wants more data); *much less* — the model
would memorise it.

**3. Byte-level BPE.**
Why: bytes as the base vocabulary mean no unknown tokens — Devanagari, Latin, digits and emoji
all encode — and frequent byte sequences become single tokens. GPT-2/3/4 and Llama 3 work this way.
Not chosen: *characters* — about 5 tokens per word, so sequences are ~4× longer and attention
cost grows with the square of length; *whole words* — Hindi inflects heavily (करना, करता,
करती, करते, करेंगे …), so the vocabulary explodes and new words become unknown; *WordPiece* —
needs an [UNK] token; *SentencePiece Unigram* — excellent, but a library: we compare against it
instead of using it.

**4. Written from scratch instead of Hugging Face `tokenizers`.**
Why: understanding the algorithm is the point of the project, and doing it ourselves is how we
found and fixed the Devanagari regex problem.
Cost: Python is slower than Rust — the incremental trainer brings training down to minutes.

**5. Devanagari-aware pre-tokenisation regex.**
Why: measured 48% fewer tokens per Hindi word than GPT-4's regex with everything else equal, and
tokens line up with real words.
Not chosen: *GPT-4's regex* — splits words at vowel signs; *no regex at all* — merges would cross
word and punctuation boundaries (tokens like "है।" or "the dog"), which generalise worse, and
training would lose the distinct-chunk speed-up.

**6. Vocabulary of 8,192 tokens.**
Why: fewer tokens per word (1.27 vs 1.45 at 4k on the dev corpus); fits in uint16; the embedding
table costs 8,192 × d parameters (≈ 3M at d = 384) — a big but reasonable share of a ~15M-parameter
model; and every token still appears often enough to be learned.
Not chosen: *4k* — longer sequences; *16k–32k* — the embedding would dominate a tiny model and rare
tokens would get too few examples; *50k (GPT-2 size)* — absurd for ~12M training tokens.

**7. NFC normalisation, not NFKC.**
Why: NFC only unifies different code-point spellings of the *same* text, so nothing visible
changes. Not chosen: *NFKC* also rewrites "compatibility" characters (ligatures, full-width
letters, superscripts) — lossy; *no normalisation* — the same word could be stored two ways.

**8. Keep ZWJ / ZWNJ, remove other invisible characters.** The joiners change how Hindi conjuncts
are drawn, so removing them changes the text; BOMs, soft hyphens and zero-width spaces don't.

**9. Merges need a pair count of at least 2.** A pair seen once teaches nothing general and would
waste a vocabulary slot on noise.

**10. Eight special tokens reserved now.** `<|endoftext|>` marks document boundaries, `<|pad|>`
pads batches, six spares are for M5's chat roles. Adding tokens later would change the vocabulary
size, and with it the shape of the model's embedding and output layers.

**11. JSONL for the cleaned corpus.** One document per line with its source: easy to stream,
inspect and filter by source. Not chosen: *plain text with separators* — loses the source, and a
separator can collide with real text; *Parquet* — compact but not human-readable.

**12. Hash-based validation split, 0.5%.** The same document lands in the same split on every
machine and in any order, and ~0.7 MB (~70k tokens) gives a stable loss estimate. Not chosen:
*a seeded random split* — changes if the input order changes; *the last N documents* — biased by
the order of the source.

**13. Exact de-duplication only.** Removes copies cheaply; near-duplicate detection (MinHash)
catches more but is a project of its own — worth adding if M3's validation loss looks too good.

**14. Script thresholds (Hindi ≥ 60% Devanagari, Hinglish ≥ 80% Latin).** Hindi pages often
contain English names and terms, so a strict 100% would drop good articles; Hinglish here means
romanised text, so lines with much Devanagari are out. These are heuristics — check them on a
sample of `data/clean/*.jsonl`.

**15. 16 Hinglish sentences per document.** Fewer `<|endoftext|>` tokens and longer training
windows. The sentences are unrelated, which is like training on shuffled sentences — fine.

**16. Tokenizer trained on a 40 MB sample.** Merge statistics settle long before 40 MB; the
hash-based sample keeps each source's share; training stays at a few minutes.

**17. uint16 token files, memory-mapped later.** 2 bytes per token instead of 8 (int64): 12M
tokens = 24 MB, and training reads random windows without loading the file (as in nanoGPT).

## 5. Interview questions you should be able to answer

1. **Why byte-level BPE rather than words or characters?** Words: huge vocabulary and unknown
   words. Characters: very long sequences. Bytes + merges: no unknowns, short sequences, small
   vocabulary.
2. **What does pre-tokenisation do?** It stops merges across word, number and punctuation
   boundaries, and lets training count distinct chunks instead of scanning the whole corpus.
3. **Why does GPT-4's tokenizer need more tokens for Hindi?** Partly English-heavy training data,
   and partly the regex: `\p{L}` excludes the combining marks Hindi uses in almost every word.
4. **How do you train BPE fast?** Distinct-chunk counts, an index from pairs to chunks, incremental
   count updates, and a heap with lazy deletion.
5. **How does encoding work?** Repeatedly merge the adjacent pair with the lowest merge rank.
6. **Why can `decode` return "�"?** A token sequence can end in the middle of a multi-byte
   character; byte-level tokens are not aligned to characters.
7. **How do you choose the vocabulary size?** Bigger vocab means fewer tokens per word, but the
   embedding and output layers grow (V × d parameters) and rare tokens get few training examples.
   8k suits a model with a few million parameters.
8. **BPE vs WordPiece vs Unigram?** BPE merges the most frequent pair; WordPiece merges the pair
   that most increases likelihood (count(ab) / (count(a)·count(b))); Unigram starts big and prunes
   the tokens whose removal costs least likelihood, and can sample segmentations.
9. **Why NFC?** One visible string can have several code-point spellings.
10. **How did you test it?** Round trips on random Unicode (Devanagari, emoji, joiners), the
    textbook example, fast trainer = naive trainer, save/load equality.

## 6. Try it yourself

- Re-implement `merge_pair` and `_encode_chunk` without looking, then run `pytest -q`.
- Train 4k, 8k and 16k vocabularies and plot Hindi tokens/word against vocabulary size.
- Open `tokenizer/tinygpt-indic-8k.vocab.txt`: what are the first 30 merges? Why those?
- Change the pattern so numbers are split into single digits; how much does fertility change?
- Time `encode_dataset.py` with the chunk cache disabled.
