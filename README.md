# TinyGPT — a Hindi & Hinglish GPT, built from scratch

A small GPT language model for **Hindi (Devanagari)** and **Hinglish (romanised Hindi-English)**,
written from first principles in Python/PyTorch: tokenizer, data pipeline, LSTM baseline,
transformer, KV-cache, fine-tuning and deployment — no `transformers` model classes.

Why Hindi? Tokenizers built for English handle Devanagari badly. GPT-style pre-tokenisation even
splits Hindi words at every vowel sign, so the same sentence costs far more tokens than in
English — less text fits in the context window and every word costs more compute.
TinyGPT's tokenizer fixes that (see [M1](docs/M1_tokenizer.md)).

## Roadmap

| | Milestone | What gets built |
|---|---|---|
| ✅ | **M1 — Data & tokenizer** | Hindi Wikipedia + Hinglish corpus; byte-level BPE from scratch with a Devanagari-aware regex; comparison with SentencePiece and GPT tokenizers |
| ⬜ | M2 — Baselines | bigram and LSTM language models; bits-per-byte evaluation |
| ⬜ | M3 — GPT from scratch | causal self-attention, pre-LN blocks, AdamW + cosine schedule; trained on an M2 MacBook (MPS) |
| ⬜ | M4 — Inference | temperature / top-k / top-p sampling, KV-cache with an equivalence test and a speed benchmark |
| ⬜ | M5 — Fine-tuning | LoRA from scratch on a Hinglish task |
| ⬜ | M6 — Ship | FastAPI service, Docker image, Gradio demo on Hugging Face Spaces |

## Quick start (M1)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.9+
pip install -e ".[data,compare,dev]"
pytest -q                                               # 39 unit tests, a few seconds

python scripts/get_data.py               # 1. corpus: Hindi Wikipedia + Hinglish (~150 MB download)
python scripts/train_tokenizer.py        # 2. BPE tokenizer, 8,192 tokens
python scripts/compare_tokenizers.py     # 3. comparison with GPT-4 regex, SentencePiece, tiktoken
python scripts/encode_dataset.py         # 4. token files for training
```

| Step | Output |
|---|---|
| 1. corpus | `data/clean/{train,val}.jsonl`, `data/clean/stats.json` |
| 2. tokenizer | `tokenizer/tinygpt-indic-8k.json` (+ readable `.vocab.txt`) |
| 3. comparison | `docs/results/tokenizers.md` |
| 4. token files | `data/tokens/{train,val}.bin` (uint16), `meta.json` |

To check the pipeline quickly without the big download, run step 1 as
`python scripts/get_data.py --source dev` (a 5 MB corpus from GitHub); the other steps are the same.

## Results (M1, full corpus)

Corpus: 134.7 MB of text → **18.2M training tokens** (Hindi Wikipedia 14.0M, Hinglish 4.1M) and
99.9k validation tokens. Tokens needed per word on held-out text (lower is better):

| Tokenizer | Vocab | Hindi tok/word | Hindi 1-token words | Hinglish tok/word |
|---|---:|---:|---:|---:|
| **TinyGPT BPE (ours)** | 8,192 | **1.65** | 65% | **1.47** |
| same BPE, GPT-4 regex | 8,192 | 2.67 | 10% | 1.34 |
| SentencePiece BPE (same data) | 8,192 | 1.67 | 66% | 1.52 |
| SentencePiece Unigram (same data) | 8,192 | 1.64 | 69% | 1.53 |
| GPT-2 | 50,257 | 8.11 | 2% | 1.69 |
| GPT-4 (cl100k) | 100,277 | 5.28 | 1% | 1.58 |
| GPT-4o (o200k) | 200,019 | 1.83 | 57% | 1.46 |
| characters | — | 5.36 | 1% | 5.14 |
| UTF-8 bytes | — | 13.64 | 1% | 5.14 |

- Same algorithm, data and vocabulary: the Devanagari-aware regex alone needs **38% fewer tokens**
  per Hindi word than GPT-4's regex.
- On par with SentencePiece (1.65 vs 1.67 / 1.64) — a from-scratch tokenizer at library quality.
- **4.9× fewer tokens than GPT-2** and **3.2× fewer than GPT-4** on Hindi, and 10% fewer than
  GPT-4o with a vocabulary 24× smaller. On Hinglish it matches GPT-4o (1.47 vs 1.46).
- Hinglish is 11% of the bytes but 23% of the training tokens: romanised text packs about
  3.5 bytes into a token, Devanagari about 8.5.

Full table with bytes/token: [docs/results/tokenizers.md](docs/results/tokenizers.md).

## Project layout

```
src/tinygpt/
  bpe.py      byte-level BPE: training (incremental pair counts + heap), encode, decode, save/load
  text.py     NFC normalisation, whitespace clean-up, script detection, train/val split
  data.py     corpus sources: Hindi Wikipedia, Hinglish datasets, small dev corpus
scripts/      get_data · train_tokenizer · compare_tokenizers · encode_dataset
tests/        39 tests: round trips on random Unicode, fast trainer == naive trainer, ...
docs/         one guide per milestone (concepts, decisions, interview questions)
```

## Data and licences

| Source | Used for | Licence |
|---|---|---|
| [Hindi Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.hi) | Hindi text | CC BY-SA 4.0 |
| [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | Hinglish sentences | Apache-2.0 |
| [Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) | Hinglish queries | Apache-2.0 |
| [UD Hindi HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB) | dev corpus only | CC BY-NC-SA 4.0 |

The data is downloaded by the scripts and never committed. Code: MIT.
