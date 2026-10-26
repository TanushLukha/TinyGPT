# TinyGPT — a Hindi & Hinglish GPT, built from scratch

A small GPT language model for **Hindi (Devanagari)** and **Hinglish (romanised Hindi-English)**,
written from first principles in Python/PyTorch: tokenizer, data pipeline, LSTM baseline,
transformer, KV-cache, fine-tuning and deployment — no `transformers` model classes.

Why Hindi? Tokenizers built for English handle Devanagari badly. GPT-style pre-tokenisation even
splits Hindi words at every vowel sign, so the same sentence costs far more tokens than in
English — less text fits in the context window and every word costs more compute.
TinyGPT's tokenizer fixes that (see [M1](docs/M1_tokenizer.md)).

**Try it:** [huggingface.co/spaces/tanushlukha/tinygpt](https://huggingface.co/spaces/tanushlukha/tinygpt) —
the model runs in your browser, nothing you type leaves it. Weights:
[huggingface.co/tanushlukha/tinygpt](https://huggingface.co/tanushlukha/tinygpt).

## Roadmap

| | Milestone | What gets built |
|---|---|---|
| ✅ | **M1 — Data & tokenizer** | Hindi Wikipedia + Hinglish corpus; byte-level BPE from scratch with a Devanagari-aware regex; comparison with SentencePiece and GPT tokenizers |
| ✅ | **M2 — Baselines** | n-gram models (counting) and an LSTM (hand-written + fused); bits-per-byte evaluation; training loop with exact resume |
| ✅ | **M3 — GPT from scratch** | GPT-2 and Llama recipes (attention by hand + fused kernel, learned positions vs RoPE); trained on the MacBook's GPU against M2's baselines |
| ✅ | **M4 — Inference** | greedy, temperature, top-k, top-p, min-p and repetition penalty measured against real text; KV-cache with equivalence tests and a speed benchmark; streaming; RoPE-only ablation |
| ✅ | **M5 — Fine-tuning** | LoRA written from scratch vs full fine-tuning vs training from scratch vs few-shot prompting, on Hinglish-TOP semantic parsing; a decontaminated base model |
| ✅ | **M6 — Ship** | safetensors bundles; a FastAPI service with streaming, backpressure and dynamic batching, load-tested; a Docker image; a public demo that runs the model in the browser (a JavaScript port checked against PyTorch) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -e ".[data,compare,dev]"
pytest -q tests/test_bpe.py tests/test_text.py         # M1's tests, a few seconds

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

**M2 — baselines** (needs PyTorch):

```bash
pip install -e ".[train]"                      # PyTorch + matplotlib
pytest -q                                      # all tests; two run only on a GPU (Apple MPS / CUDA)

python scripts/ngram_baselines.py              # 5. n-gram baselines, about a minute
python scripts/bench.py                        # 6. how fast does training run on this machine?
caffeinate -i python scripts/train.py          # 7. LSTM baseline (caffeinate keeps a Mac awake)
python scripts/sample.py                       # 8. generate text from the trained LSTM
python scripts/report.py                       # 9. one table with every baseline + learning curves
python scripts/compare_models.py               # 10. per-source scores + LSTM/n-gram mixture
```

| Step | Output |
|---|---|
| 5. n-grams | `docs/results/m2_ngram.md` (+ `.json`) |
| 7. LSTM | `checkpoints/lstm/`: `best.pt`, `last.pt`, `metrics.jsonl`, `results.json` |
| 9. report | `docs/results/m2_baselines.md`, `docs/results/m2_curves.png` |
| 10. comparison | `docs/results/compare.md` |

**M3 — the GPT:**

```bash
python scripts/bench.py --model gpt            # 11. how long will a GPT run take on this machine?
caffeinate -i bash scripts/run_m3.sh           # 12. three GPTs, then tables, curves, attention maps, samples
```

`run_m3.sh` trains `gpt` (GPT-2 recipe, 13.9M parameters), `gpt-small` (5.3M, the LSTM's size) and
`gpt-llama` (RoPE, RMSNorm, SwiGLU), then writes `docs/results/m3_baselines.md`, `m3_curves.png`,
`m3_compare.md`, `m3_context.png`, `attention.png` and `attention_llama.png`. Re-running it skips
finished runs and resumes an interrupted one.

**M4 — inference:**

```bash
caffeinate -i bash scripts/run_m4.sh           # 13. speed + decoding benchmarks, the RoPE-only GPT, head census
python scripts/sample.py --stream --top-p 0.9 --top-k 0      # text appears as it is generated
```

`run_m4.sh` writes `docs/results/m4_generation.md` + `.json` + `m4_latency.png` (time per token
with and without the KV-cache), `m4_decoding.md` (8 decoding strategies vs real text), trains `gpt-rope`
(GPT-2 recipe with RoPE only), then `m4_rope_baselines.md`, `m4_rope_curves.png`,
`m4_rope_compare.md`, `m4_rope_context.png` and `heads.md` + `heads.png` (attention-head census).

**M5 — fine-tuning:**

```bash
caffeinate -i bash scripts/run_m5.sh           # 14. clean base model, ~30 fine-tuning runs, report (~3 h)
python scripts/parse.py --out-dir runs/m5-annotated --run full-long "kal subah 7 baje ka alarm laga do"
```

`run_m5.sh` downloads Hinglish-TOP (`data/top/`), removes its validation and test requests from the
pretraining text (`data/clean_dc/`, `data/tokens_dc/`), pretrains `gpt-llama-clean`, runs the
fine-tuning experiments (`runs/m5/`, `runs/m5-annotated/`) and writes `docs/results/m5_finetune.md`,
`m5_rank.png` and `m5_data.png`. It prints the output format and learning rates it chose and the
name of the best LoRA run.

**M6 — ship:**

```bash
pip install -e ".[serve,deploy]"               # FastAPI, uvicorn, Gradio, safetensors, Prometheus client
bash scripts/run_m6.sh                         # 15. export both models, every test, load test (~10 min)
python -m tinygpt.serve                        # 16. API + demo on http://127.0.0.1:8000 (API docs: /docs)
docker build -t tinygpt . && docker run --rm -p 8000:8000 tinygpt     # 17. the same, in a container
python -m http.server 8000                     # 18. the browser demo: http://localhost:8000/web/
hf auth login && python scripts/deploy_hf.py   # 19. model repo + the browser demo as a static Space
```

```bash
curl -s localhost:8000/v1/parse -H 'content-type: application/json' \
     -d '{"text": "kal subah 7 baje ka alarm laga do"}'
curl -N localhost:8000/v1/generate/stream -H 'content-type: application/json' \
     -d '{"prompt": "भारत की राजधानी", "max_new_tokens": 60}'
```

`run_m6.sh` writes `models/lm/`, `models/parser/` (safetensors bundles, checked against the checkpoints
and against M5's decoding) and `models/README.md` (the model card), then `docs/results/m6_facts.md`
(what the language model knows), `m6_load.md`, `m6_load.json`, `m6_parse.png` and — with Node.js
installed (`brew install node`) — `m6_web.md` (the browser version against PyTorch). Docker and the
browser demo need `models/` from step 15; `deploy_hf.py` uploads it (`--kind zerogpu` deploys the
Gradio app instead, for accounts with ZeroGPU). `node web/test/e2e.mjs` checks the demo page in
Chromium (once: `npm install --no-save playwright && npx playwright install chromium`).

An interrupted run continues exactly where it stopped with `python scripts/train.py --resume`.
Every training setting is a flag (`python scripts/train.py --help`).

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

## Results (M2, baselines)

Validation bits per byte — total cross-entropy divided by the size of the text, so it doesn't
depend on the tokenizer (lower is better; details in [M2](docs/M2_baselines.md)):

| Model | Size | Bits/byte | Hindi | Hinglish |
|---|---|---:|---:|---:|
| uniform (1/8,192 per token) | — | 1.781 | 1.572 | 3.738 |
| unigram | 8k counts | 1.482 | 1.303 | 3.166 |
| bigram | 1.7M counts | 0.983 | 0.915 | 1.617 |
| trigram | 8.9M counts | 0.877 | 0.838 | 1.239 |
| 4-gram | 20.6M counts | 0.848 | 0.829 | **1.029** |
| **LSTM**, 2 × 384 (16 min on the M2's GPU) | 5.5M parameters | **0.831** | **0.785** | 1.265 |
| LSTM + 4-gram mixture | both | 0.766 | 0.740 | 1.016 |

![M2 learning curves](docs/results/m2_curves.png)

- The LSTM beats every n-gram overall and on Hindi with a quarter of the stored numbers, and was
  still improving when its 32.8M-token budget ran out.
- The 4-gram wins on Hinglish: its validation sentences mostly repeat training phrases (only 2.4%
  of Hinglish-TOP's 4-grams are new), which exact counts replay perfectly — memorisation vs
  generalisation.
- Mixing the two gives 0.766: they know different things. That is the bar for M3's GPT.

## Results (M3, GPT from scratch)

Same data, same 32.8M training tokens, same evaluation as M2 (details in [M3](docs/M3_gpt.md)):

| Model | Size | Bits/byte | Hindi | Hinglish | Time on the M2 GPU |
|---|---|---:|---:|---:|---:|
| 4-gram (M2) | 20.6M counts | 0.848 | 0.829 | **1.029** | — |
| LSTM (M2) | 5.5M | 0.831 | 0.785 | 1.265 | 16 min |
| GPT-2 recipe, 4 × 256 | 5.3M | 0.823 | 0.770 | 1.319 | 18 min |
| GPT-2 recipe, 6 × 384 | 13.9M | 0.784 | 0.735 | 1.241 | 50 min |
| **Llama recipe, 6 × 384** (RoPE, RMSNorm, SwiGLU) | 13.8M | **0.733** | **0.689** | 1.147 | 53 min |

![M3 learning curves](docs/results/m3_curves.png)

- At the LSTM's size, attention wins by only 1%; scale (2.6× parameters) gives 4.7%; the 2023
  recipe another **6.5% at the same size** — 12% better than the LSTM, 14% better than the 4-gram.
- Only the RoPE model keeps using context beyond ~64 tokens, and only it has started to copy a
  repeated random sequence: its heads form the classic previous-token → induction circuit (72% of
  one head's attention goes to the previous token; another head puts 23% on the copy target).
- The 4-gram still wins on templated Hinglish; mixing it with the best GPT gives 0.701.

## Results (M4, inference)

Details in [M4](docs/M4_inference.md).

**Was M3's gain RoPE?** The GPT-2 recipe with *only* its positions swapped for RoPE:

| Model | Bits/byte | Hindi | Hinglish | Previous-token head | Induction head |
|---|---:|---:|---:|---:|---:|
| GPT-2 recipe | 0.784 | 0.735 | 1.241 | 31% | 3% |
| **GPT-2 recipe + RoPE** | 0.737 | 0.691 | 1.168 | 75% | 18% |
| Llama recipe (RoPE, RMSNorm, SwiGLU, no biases) | 0.733 | 0.689 | 1.147 | 72% | 23% |

RoPE alone gives 91% of M3's gain (5.9% of the 6.5%) and grows the same previous-token → induction
circuit; RMSNorm, SwiGLU and dropping biases add the last 0.6%.

**Decoding** (gpt-llama; 10 validation prompts, 80 new tokens; nll = how predictable the text is to
the model, repeat-4 = share of 4-token sequences that repeat earlier ones):

| Strategy | nll | distinct-2 | repeat-4 |
|---|---:|---:|---:|
| real text (reference) | 2.96 | 0.95 | 0.00 |
| greedy | 0.99 | 0.25 | 0.73 |
| temperature 1.0 | 2.85 | 0.94 | 0.01 |
| top-p 0.9 | 2.14 | 0.88 | 0.03 |
| min-p 0.1 | 1.59 | 0.83 | 0.05 |

- Greedy text is the most likely and the worst: 73% of its 4-token sequences are repeats.
- Pure sampling matches real text on all three numbers yet invents words. The model's tail isn't
  over-weighted — the real next token falls in it about as often as its probability says — the model just
  can't tell which rare token is right, so cutting the tail (top-p, min-p) trades fidelity for safety.

**Speed** (MacBook Air M2, 240 new tokens, tokens per second):

| Model | CPU | GPU (MPS) | GPU, 16 sequences at once |
|---|---:|---:|---:|
| gpt-llama, no cache | 108 | 140 | 335 |
| gpt-llama, KV-cache | 550 | 274 | 3,415 |
| LSTM | 3,201 | 882 | 12,406 |

![time per token](docs/results/m4_latency.png)

- The KV-cache keeps the time per token flat instead of growing with the context: 5× faster on the
  CPU over 240 tokens, with tests proving it generates exactly the same text.
- For one sequence the CPU beats the GPU (550 vs 274): each step is ~500 tiny operations plus a wait
  for the GPU's answer, a fixed price that dominates. With 16 sequences the GPU catches up (3,415 vs
  3,314 on the CPU).

## Results (M5, fine-tuning)

Hinglish-TOP: turn an assistant request into its intent and slots, e.g. `kal subah 7 baje ka alarm
laga do` → `[IN:CREATE_ALARM [SL:DATE_TIME kal subah 7 baje ] ]`. 2,993 training examples; exact match
of the whole parse on 6,513 test requests (details in [M5](docs/M5_lora.md)):

| Method | Trainable weights | To store per task | Test exact match | Slot F1 |
|---|---:|---:|---:|---:|
| lookup table (identical training request) | 0 | — | 9.3% | 11.0% |
| few-shot: 3 solved examples in the prompt | 0 | — | 0.0% | 0.0% |
| same network, from scratch | 13.8M | 55 MB | 19.1% | 28.1% |
| **full fine-tuning** | 13.8M | 55 MB | **38.0%** | **53.4%** |
| LoRA r=8 (written from scratch) | 313K (2.3%) | 1.3 MB | 19.5% | 29.9% |
| *3× longer training:* full fine-tuning / LoRA | | | 41.0% / 32.4% | 56.8% / 46.7% |

![exact match vs trainable weights](docs/results/m5_rank.png)

- At 14M parameters, full fine-tuning wins; LoRA mostly learns slower — 3× the steps take it from
  19.5% to 32.4% (full fine-tuning: 38.0% → 41.0%). Rank matters (r=1: 3.1%, r=64: 25.9%), and so do
  the MLP adapters (attention-only: 5.1%).
- Pretraining doubles the parser (38.0% vs 19.1% from scratch); the 14M model can't learn from
  examples in its prompt at all.
- 99.5% of the test requests were in M1's pretraining text. A base model retrained without them
  parses exactly as well (LoRA 19.5% vs 19.2%, full 38.0% vs 37.5%), although the old one had
  partly memorised them (0.235 nats/token lower loss on them).
- Failure mode: names the training data never had are replaced by ones it had (`Delhi` → `Duran`).

## Results (M6, shipping)

Details in [M6](docs/M6_serving.md). The parser behind the API (MacBook Air M2, client and server on the
same machine; requests per second and median latency):

| Clients | Batching off | Batching on | Mean batch | Median latency, off → on |
|---:|---:|---:|---:|---:|
| 1 | 12.4 | 12.3 | 1.0 | 72 → 72 ms |
| 8 | 13.0 | 17.6 | 6.2 | 611 → 408 ms |
| 32 | 12.7 | **32.0** | 16.0 | 2,512 → **912** ms |

![parse throughput and latency](docs/results/m6_parse.png)

- Parse requests that wait together share one forward pass: **2.5× the throughput at 32 clients** at
  a third of the latency, and no change for a single client (the batcher never waits for company).
- Generation streams ~455 tokens/s, one stream at a time; a full queue answers 503 in 22 ms instead
  of letting waits grow; a client that disconnects stops its generation.
- **The browser demo is the same model:** the JavaScript port gives the same token ids on all 5,503
  validation texts and requests, logits within 3.8 × 10⁻⁶ of PyTorch, the same greedy text (10 of 10)
  and the same parse trees (200 of 200) — at ~35 ms per token on one CPU thread.
- **It learned the language, not the facts:** asked for the capitals of 20 states and countries, it
  ranks the right city first twice — chance level — and the demo says so.

## Project layout

```
src/tinygpt/
  bpe.py        byte-level BPE: training (incremental pair counts + heap), encode, decode, save/load
  text.py       NFC normalisation, whitespace clean-up, script detection, train/val split
  data.py       corpus sources: Hindi Wikipedia, Hinglish datasets, small dev corpus
  dataset.py    memory-mapped token files: random training windows, exact evaluation windows
  metrics.py    bits per byte, overall and per language
  plotting.py   one look for every chart, a fixed colour per model
  trainer.py    training loop: AdamW, warm-up + cosine, clipping, checkpoints, exact resume
  sampling.py   greedy, temperature, top-k, top-p, min-p, repetition penalty; streaming text
  lora.py       LoRA: a low-rank update for any linear layer; merge, save and load adapters
  top.py        Hinglish-TOP: parse trees, the decoupled form, scores, prompts, batched greedy parsing
  bundle.py     model bundles: safetensors weights + config + tokenizer + checksums
  models/
    ngram.py    n-gram counts as sorted int64 codes; add-k and Jelinek-Mercer smoothing
    lstm.py     LSTM language model; ScratchLSTM = the same LSTM written gate by gate
    gpt.py      GPT: GPT-2 / Llama recipes, attention by hand or fused, RoPE, RMSNorm, SwiGLU,
                KV-cache, left-padded batches of different lengths
  serve/
    engine.py   the two models as plain objects: streaming generation, batched parsing
    worker.py   one thread per model: bounded queue (503), cancelling, dynamic batching
    api.py      FastAPI: JSON + Server-Sent Events endpoints, validation, request ids, metrics
    ui.py       the service's Gradio demo (also the ZeroGPU Space's page)
web/            the public demo, a static page: tinygpt.js (tokenizer, model, sampling and parser in
                JavaScript), worker.js (downloads, checks and runs the models off the page's thread),
                app.js + index.html + style.css (the page), test/ (the Python tests' bridge to the
                JavaScript; e2e.mjs: the page in a real browser)
scripts/        get_data · train_tokenizer · compare_tokenizers · encode_dataset ·
                ngram_baselines · bench · train · sample · report · compare_models · attention ·
                heads · decoding · bench_generate · decontaminate · finetune · m5_best · report_m5 ·
                parse · export · probe_facts · load_test · check_web · deploy_hf · run_m3.sh · run_m4.sh ·
                run_m5.sh · run_m6.sh
space/          the Gradio app for a ZeroGPU Space: app.py, requirements, settings
Dockerfile      the service image: CPU-only PyTorch, pinned dependencies (requirements/), non-root
tests/          round trips on random Unicode, fast BPE == naive BPE, hand-written LSTM ==
                nn.LSTM, n-gram scores == full distributions, stop-and-resume == uninterrupted,
                KV-cache == full forward pass, padded batch == each row alone, streamed text ==
                decoded text, LoRA starts as the base model and merges exactly, fine-tuning runs end
                to end, bundles load exactly, the API's limits, streams, 503s and batches, the
                JavaScript port == the Python code, what the deploy would upload, ...
docs/           one guide per milestone (concepts, decisions, interview questions)
```

## Data and licences

| Source | Used for | Licence |
|---|---|---|
| [Hindi Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.hi) | Hindi text | CC BY-SA 4.0 |
| [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | Hinglish sentences | Apache-2.0 |
| [Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) | Hinglish queries; M5's parsing task | Apache-2.0 |
| [UD Hindi HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB) | dev corpus only | CC BY-NC-SA 4.0 |

The data is downloaded by the scripts and never committed. Code: MIT.
