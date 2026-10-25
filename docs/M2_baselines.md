# M2 — Baselines: n-grams and an LSTM

**Goal:** before building the GPT, measure how well *simpler* models do on exactly the same data
and evaluation. A number like "1.2 bits per byte" means nothing on its own; "the GPT is 25% better
than the best LSTM and 40% better than a 4-gram" does. Along the way M2 builds the training,
evaluation and checkpoint code that M3 reuses unchanged.

```
data/tokens/train.bin ─┬─ count n-grams ──── uniform · unigram · bigram · trigram · 4-gram
                       └─ trainer.py ───────── LSTM ── checkpoints/lstm/{best,last}.pt
data/tokens/val.bin ───── metrics.py: every validation token, scored in bits per byte
```

## 1. The task and the loss

A language model gives the probability of the next token given the ones before it,
p(x_t | x_1 … x_{t-1}). Training minimises the **cross-entropy**: the average of −ln p(correct
next token), in nats per token.

- **Perplexity** = e^loss: the model is "as unsure as if it picked uniformly among that many
  tokens". A model that knows nothing gives every one of the 8,192 tokens probability 1/8,192:
  loss ln 8192 = **9.01**, perplexity 8,192.
- That is also the loss a freshly initialised model must start at. If the first evaluation shows
  15, the initialisation is broken — `tests/test_lstm.py::test_initial_loss_is_log_vocab` checks it.

## 2. Measuring: bits per byte — `metrics.py`

```
bits per byte = (sum of −ln p over all predicted tokens) / (ln 2 × number of UTF-8 bytes of the text)
```

**Why not loss or perplexity per token?** They depend on the tokenizer. A tokenizer with longer
tokens makes fewer, harder predictions, so its per-token loss is higher even if the model is
better. Dividing the total information by the size of the *text* puts any tokenizer — ours,
GPT-2's, characters, raw bytes — on one scale.

**It also makes the languages comparable.** The uniform model pays 13 bits for every token, but a
Hindi token covers 8.3 bytes and a Hinglish one 3.5, so per byte it costs **1.57** on Hindi and
**3.74** on Hinglish. Every result below is reported overall and per language.

**Prediction = compression.** With arithmetic coding, a model that assigns X bits per byte can
compress text to X/8 of its size. For reference, `ngram_baselines.py` also runs gzip, bzip2 and xz
on the validation text. (They only see that text itself; our models learned from 18M tokens.)

**The evaluation protocol** (identical for every model in this repo):
- all 99,856 validation tokens, no sampling — the set is small enough to score exactly in seconds;
- each source (Hindi Wikipedia, two Hinglish sets) is scored separately: in its slice of `val.bin`,
  every token except the first is predicted exactly once (`tests/test_dataset.py` checks this);
- neural models read the slice in non-overlapping windows of 256 tokens, and every window starts
  with no memory, so on average a token is predicted from ~128 tokens of context;
- `<|endoftext|>` is predicted like any token (the model must learn where documents end); it adds
  no bytes.

## 3. n-gram models — `models/ngram.py`, `scripts/ngram_baselines.py`

An n-gram model predicts a token from the n−1 tokens before it, using training counts:

```
p_ML(w | h) = c(h, w) / c(h)        "of the times context h appeared, how often did w follow?"
```

**Sparsity is the whole problem.** There are 67M possible bigrams and 550 billion possible trigrams
over our vocabulary; 18M training tokens contain 1.7M distinct bigrams, 7.2M trigrams and 11.7M
4-grams. Of the validation set's 4-grams, **58% never occur in training** (trigrams 33%, bigrams
6%). With raw counts each of those gets probability 0, i.e. infinite loss. **Smoothing** moves
some probability to unseen events:

| method | formula | idea |
|---|---|---|
| add-k | (c(h,w) + k) / (c(h) + kV) | pretend every token followed h k extra times |
| Jelinek-Mercer (JM) | λ·p_ML(w \| h) + (1−λ)·p_{n−1}(w \| shorter h) | mix in the next-shorter context's estimate, recursively down to uniform; if h was never seen, use the shorter context alone |

The weights (k, λ) are picked on the validation set, one order at a time.

**Implementation.** Each n-gram is coded as one int64 (t₁·V^(n−1) + … + t_n), the codes are
sorted with `np.unique`, and a lookup is a binary search (`np.searchsorted`): counting 1- to 4-grams
of 18M tokens takes ~8 s and ~1.5 GB of RAM, and all validation lookups are vectorised. A Python
dict of tuples would need several GB and minutes; a dense array only works for bigrams (8,192² =
67M cells) — trigrams would need 550 billion.

**Results** (validation; lower is better):

| model | smoothing | n-grams stored | loss/token | perplexity | bits/byte | Hindi | Hinglish |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | 1/V | — | 9.011 | 8,192 | 1.781 | 1.572 | 3.738 |
| unigram | JM | 8,125 | 7.499 | 1,807 | 1.482 | 1.303 | 3.166 |
| bigram | add-k (k = 0.03) | 1.7M | 4.980 | 146 | 0.984 | 0.917 | 1.612 |
| bigram | JM | 1.7M | 4.973 | 144 | 0.983 | 0.915 | 1.617 |
| trigram | JM | 8.9M | 4.437 | 85 | 0.877 | 0.838 | 1.239 |
| 4-gram | JM | 20.6M | **4.291** | 73 | **0.848** | 0.829 | 1.029 |

**Add-k gets worse with more context:** an add-k trigram scores 1.045 bits/byte, worse than the
add-k bigram (0.984). With 8,192 possible next tokens, the fake counts swamp the few real ones in
rare contexts, and an unseen context gets the uniform distribution. Interpolation can always lean on
the shorter context (small λ), so adding an order never hurts much.

**Compared with general-purpose compressors** on the same validation text (§2):

| compressor | bits/byte | Hindi | Hinglish |
|---|---:|---:|---:|
| gzip -9 | 1.773 | 1.645 | 2.908 |
| xz -9e | 1.450 | 1.320 | 2.685 |
| bzip2 -9 | 1.293 | 1.154 | 2.522 |

A bigram model (0.98) already beats all three — the compressors only see the 730 KB they compress,
the models bring statistics from 134 MB of training text.

- **One token of context is worth the most:** unigram → bigram cuts 34% of the bits.
- **Then returns shrink:** bigram → trigram −11%, trigram → 4-gram −3%, because longer contexts are
  mostly unseen. Counting cannot generalise — "राजधानी दिल्ली" teaches nothing about
  "राजधानी मुंबई". Models that represent tokens as vectors (embeddings) can share what they learn
  between similar contexts; that is what the LSTM adds.
- **Hinglish gains most from longer contexts** (1.62 → 1.03 from bigram to 4-gram): much of it is
  short, templated commands ("set an alarm for 6 am") from Hinglish-TOP.
- The 4-gram stores 20.6M counts — four times the LSTM's parameters.

Samples (`docs/results/m2_ngram.md`) are fluent over three or four tokens and nonsense after that:
> yaar kal movie dekhne he / tuesday ko 645 pm me weather mil sakta hai

## 4. The LSTM — `models/lstm.py`

**Why recurrent networks needed fixing.** A plain RNN updates h_t = tanh(W x_t + U h_{t−1}). The
gradient from step t back to step t−k is a product of k Jacobians (U times the tanh slope): it
shrinks to nothing (vanishing) or blows up (exploding) as k grows, so the network can't learn
anything that spans more than a few steps.

**The LSTM's fix** is a separate cell c that is updated by *addition*:

```
i = σ(W_i x_t + U_i h + b_i)    input gate:  how much of the candidate to write
f = σ(W_f x_t + U_f h + b_f)    forget gate: how much of the old cell to keep
g = tanh(W_g x_t + U_g h + b_g) candidate values
o = σ(W_o x_t + U_o h + b_o)    output gate: how much of the cell to show
c_t = f ⊙ c_{t−1} + i ⊙ g       ∂c_t / ∂c_{t−1} = f — close to 1 whenever the gate says "keep"
h_t = o ⊙ tanh(c_t)
```

Along the cell, the gradient is multiplied by f (≈ 1 when remembering) instead of by a weight
matrix, so it survives many steps. Exploding gradients are still possible, which is why training
clips the gradient norm (§5).

**Our model:** tokens → embedding (8,192 × 384) → 2 LSTM layers (384 units) → output layer that
*reuses the embedding matrix* (weight tying) → 8,192 logits. 5.51M parameters, of which 2.37M are
not the embedding.

**Two implementations, one set of weights.** `ScratchLSTM` writes the equations above out by hand:
the input part of all 256 steps is one matrix multiply, then a Python loop over time does the
recurrent part — it can't be parallelised over time, because step t needs h_{t−1}. That loop
launches thousands of tiny GPU operations per batch, so training uses PyTorch's fused `nn.LSTM`
(Metal kernels on a Mac). `ScratchLSTM` stores its parameters under `nn.LSTM`'s names and in its
layout (gates stacked i, f, g, o), so a checkpoint loads into either, and
`test_scratch_lstm_matches_pytorch` checks that outputs, final states and gradients agree.

**Initialisation — and a bug the curves caught.** The embedding was first initialised like GPT-2's,
N(0, 0.02). Short runs on the real data sat on a plateau just above the *unigram* loss (7.5–7.9
nats) for hundreds of steps. Experiments (300 steps each, CPU) isolated the cause:

| variant | val loss after 100 / 200 / 300 steps |
|---|---|
| tied, N(0, 0.02) | 7.83 / 7.67 / 7.55 |
| tied, N(0, 0.02), input scaled by √384 | 7.82 / 7.67 / 7.55 |
| untied: input N(0, 1), output N(0, 0.02) | 7.76 / 7.61 / 7.53 |
| **tied, U(−0.1, 0.1)** | **7.46 / 7.05 / 6.77** |

Scaling up the *input* changed nothing and small *output* weights were slow even untied, so the
problem was on the output side. Our explanation, partly measured: at every step the softmax lowers
the score of every token that wasn't next, so every row of the output matrix gets the same small
push — and Adam makes each push about the size of the learning rate however small the gradient is.
Rows that start at ±0.02 are soon dominated by that shared push and begin to look alike: after 60
steps, the average cosine similarity between rare tokens' vectors was 0.21 (0.08 with
U(−0.1, 0.1)), mid-frequency ones 0.11 (0.04). Output vectors that look alike send the LSTM a
blurry error signal about *which* token should come next, so the context takes hundreds of steps to
start helping. U(−0.1, 0.1) is what PyTorch's `word_language_model` and AWD-LSTM use; GPT-style
models use 0.02 with learning rates 10× smaller and long warm-ups, so the shared push takes far
longer to swamp the starting values.

The **forget-gate bias starts at 1**: with a bias of 0 the gate starts at σ(0) = 0.5, halving the
cell at every step, so memories fade within a few tokens; at 1 it starts at 0.73.

**Generation is cheap per token:** the prompt is read once, then each new token is one LSTM step
from the carried (h, c) — the state is a fixed-size summary of everything so far. (Without a cache,
a GPT re-reads the whole context for every token; M4's KV-cache fixes that.) The flip side: that
fixed-size summary is also why an LSTM forgets.

## 5. Training — `trainer.py`, `scripts/train.py`

| | setting | why |
|---|---|---|
| batches | 32 random windows × 256 tokens = 8,192 tokens | random windows from the memory-mapped file: no epoch bookkeeping, nothing loaded into RAM |
| length | 4,000 steps = 32.8M tokens ≈ 1.8 epochs | a fixed token budget; M3's GPT gets the same |
| optimiser | AdamW, peak lr 1e-2, β = (0.9, 0.99), weight decay 0.01 on matrices only | per-parameter step sizes cope with rare tokens' sparse gradients |
| schedule | 100 warm-up steps, cosine decay to 1e-3 | warm-up: Adam's early statistics are noisy; decay: settles into a lower loss |
| clipping | gradient norm ≤ 1.0 | one exploding batch can't wreck the weights |
| dropout | 0.1 on embeddings, between layers and before the output | mild regularisation for a second epoch |
| evaluation | every 250 steps, exact, on all validation tokens | the curve has no sampling noise |

**The learning rate was measured, not guessed** (1,200 steps of 8 × 256 tokens on CPU, cosine
schedule over those steps, U(−0.1, 0.1) init):

| peak learning rate | val loss after 400 / 800 / 1,200 steps |
|---|---|
| 3e-3 | 5.93 / 5.52 / 5.39 |
| **1e-2** | **5.69 / 5.31 / 5.14** |

1e-2 led at every evaluation, and the gradient norm stayed around 0.15 — no sign of instability.
The full run uses 4× bigger batches, whose less noisy gradients tolerate a high rate at least as
well. (Each architecture gets its own tuned rate: M3's GPT will need a much smaller one.)

**Checkpoints.** `best.pt` keeps the weights with the lowest validation loss; `last.pt` keeps
everything needed to continue *exactly*: weights, AdamW's moment estimates, the step (which fixes
the learning rate), and the random-number states that choose batches and dropout masks.
`python scripts/train.py --resume` continues an interrupted run, and
`test_resume_is_exact` checks that stop-and-resume gives bit-for-bit the same model as never
stopping. Files are written to a temporary name and renamed, so a crash mid-save can't corrupt the
previous checkpoint, and they are loaded with `weights_only=True`, so loading can't run code.

**Speed.** `loss.item()` makes the CPU wait for the GPU to finish everything queued. Calling it every
step stalls the pipeline, so the loop adds losses up on the GPU and reads them every 50 steps.

## 6. Results

The LSTM trained for 4,000 steps (32.8M tokens, 1.8 epochs) in **16 minutes on the M2's GPU**
(~35k tokens/s; `bench.py` measured 39k on the GPU vs 15k on the CPU).

| model | size | loss/token | bits/byte | Hindi | Hinglish |
|---|---|---:|---:|---:|---:|
| uniform | — | 9.011 | 1.781 | 1.572 | 3.738 |
| unigram | 8.1k counts | 7.499 | 1.482 | 1.303 | 3.166 |
| bigram (JM) | 1.7M counts | 4.973 | 0.983 | 0.915 | 1.617 |
| trigram (JM) | 8.9M counts | 4.437 | 0.877 | 0.838 | 1.239 |
| 4-gram (JM) | 20.6M counts | 4.291 | 0.848 | 0.829 | **1.029** |
| **LSTM**, 2 × 384 | 5.5M parameters | **4.206** | **0.831** | **0.785** | 1.265 |
| LSTM + 4-gram, mixed 0.6 / 0.4 | both | — | 0.766 | 0.740 | 1.016 |

![learning curves](results/m2_curves.png)

**Reading the curves.** The LSTM passes the bigram after ~5M training tokens and the 4-gram after
~22M, and ends 2% better than the 4-gram overall and 5% better on Hindi — with a quarter as many
stored numbers (5.5M vs 20.6M). Validation loss tracks training loss the whole way, and the best
checkpoint is the *last* one: the model was still improving, so it was limited by training time,
not by data. M3's GPT gets the same 32.8M-token budget, so the comparison stays fair.

**But the 4-gram wins on Hinglish.** Broken down by source (`scripts/compare_models.py`):

| source | 4-grams never seen in training | 4-gram | LSTM |
|---|---:|---:|---:|
| Hindi Wikipedia | 68.6% | 0.829 | **0.785** |
| findnitai (Hinglish sentences) | 24.6% | **1.233** | 1.435 |
| Hinglish-TOP (assistant commands) | 2.4% | **0.746** | 1.029 |

This is **memorisation vs generalisation**. Almost every phrase in Hinglish-TOP's validation
sentences already occurs in training ("… ke liye alarm set kare" alone appears 5,573 times), and a
table of 20.6M exact counts is the perfect tool for replaying them. The LSTM squeezes everything
into 2.4M recurrent weights, and with Hinglish only 23% of its training tokens, it spends that
capacity on Hindi — where 69% of the 4-grams are new and only generalisation helps. The order of
the gaps follows the unseen column exactly.

**The two models know different things.** Mixing their probabilities for every token
(0.6 × LSTM + 0.4 × 4-gram) gives **0.766 bits/byte**, 8% better than the LSTM alone. That is the
real bar for M3: a GPT should beat the mixture on its own.

**Samples** (`scripts/sample.py`, temperature 0.8, top-k 50):

> महात्मा गांधी का जन्म 12 फ़रवरी 1933), (कभी-कभी सोपकी) - ब्रिटिश वंश के संस्थापक थे। वे सन् 1947 में
> बनी हिन्दी भाषा की फिल्म है। यह फिल्म 2007 में बनी थी।

> yaar kal movie dekhne ke liye reminder set kare / monday subah 8 baje ke liye alarm set kare /
> Aaj raat mere aas paas konse events chal rahe he

The Hindi is grammatical and sounds like Wikipedia, but the facts are invented and the subject
changes every sentence (Gandhi → founder of a British dynasty → a 1947 film): a fixed-size memory
keeps the *style* of the text, not its *topic*. The Hinglish turns into the assistant commands it
trained on. Some samples also reproduce a quirk of the Hinglish-TOP data — pre-tokenised spacing
("I ' m on my way .") and literal "\n" — which the model learned faithfully (see decision 23).

## 7. Decisions we made, and why

Each entry: what we chose → why → the alternatives and why we didn't pick them.

**1. Baselines before the GPT.** Why: without a reference, M3's loss is uninterpretable, and the
two baselines answer different questions — n-grams: how much is plain local statistics? LSTM: what
does attention add over the best older architecture? Not chosen: *comparing with published
numbers* — different data and tokenizer make them meaningless; *skipping baselines* — then "it
works" has no evidence.

**2. Bits per byte as the headline metric.** Why: independent of the tokenizer, comparable across
Hindi and Hinglish, and directly the compressed size of the text. Not chosen: *perplexity per
token* — changes with the tokenizer; *bits per character* — also fine, but bits per byte is the
convention (enwik8, The Pile) and needs no Unicode decisions.

**3. Exact evaluation on all validation tokens.** Why: 100k tokens score in seconds, so there's no
reason to estimate; every curve point and the final number come from the same function. Not
chosen: *a few random batches* (nanoGPT's way) — needed only when validation is large.

**4. Non-overlapping 256-token windows, each starting fresh.** Why: cheap and identical for every
model, including M3's GPT, whose context is 256. Not chosen: *sliding windows* — every token gets
near-full context and lower loss, but costs 256/stride times more compute (worth adding for M3's
final numbers); *carrying the LSTM state across windows* — would give the LSTM context the GPT
can't have.

**5. No separate test set.** Why: validation is only 0.5% of the data; splitting it again would make
both halves noisy (the Hinglish part is ~20k tokens). Choosing k, λ and the best checkpoint on it
flatters every model a little and in the same way, and the conclusions are comparisons between
models. Not chosen: *a three-way split* — the honest choice with more data; worth doing if numbers
are ever published.

**6. n-grams stored as sorted int64 codes.** Why: 20.6M counts in a few hundred MB, built in seconds,
looked up with vectorised binary search. Not chosen: *Python dicts* — several GB and slow; *dense
arrays* — only possible for bigrams; *KenLM* — the standard C++ toolkit, but a library, and the
point is to understand it.

**7. Add-k and Jelinek-Mercer smoothing; orders up to 4.** Why: add-k is the textbook first idea and
shows why it fails (too much mass to the thousands of unseen continuations); JM interpolation is a
few lines and works well. Order 4 because gains were already down to 3% and 5-gram codes overflow
int64 at V = 8,192. Not chosen: *Kneser-Ney* — the best classic smoother (it asks "in how many
contexts does w appear?" rather than "how often?"), left as an exercise.

**8. An LSTM as the neural baseline.** Why: the strongest architecture before Transformers (AWD-LSTM
era), and the cleanest contrast with attention: a fixed-size memory updated step by step versus
direct access to every previous token. Not chosen: *a plain RNN* — vanishing gradients make it a
straw man; *a GRU* — similar to the LSTM with fewer gates, but the LSTM is the standard reference;
*an MLP over a fixed window* (Bengio 2003) — weaker, and no memory at all.

**9. 2 layers of 384 units, 5.5M parameters.** Why: the same width as M3's GPT, and a run that fits
in well under an hour on a fanless MacBook Air. Two layers is the usual depth for LSTM language
models; deeper ones gain little and each layer adds sequential work. Not chosen: *a GPT-sized LSTM*
(~10M non-embedding parameters) — 3–4× slower per step; the parameter counts are reported next to
every result instead.

**10. Weight tying.** Why: the input embedding and the output layer both map between tokens and the
same 384-dimensional space; sharing them saves 3.1M parameters (otherwise more than half the model)
and improves perplexity (Press & Wolf 2017, Inan et al. 2017). GPT-2 does the same. It requires the
LSTM's output size to equal the embedding size.

**11. Embedding initialised U(−0.1, 0.1).** Why: measured — see §4: N(0, 0.02) left training on a
plateau near the unigram loss for hundreds of steps. Not chosen: *N(0, 0.02)* — GPT-2's choice,
but GPT-style training uses much smaller learning rates; *scaling the input by √d* (the Transformer
trick) — tested, no effect, because the bottleneck was on the output side.

**12. Forget-gate bias 1.** Why: the gate starts mostly open, so the cell keeps information (and
gradients) over many steps from the first update (Gers et al. 2000; Jozefowicz et al. 2015). Not
chosen: *0* (PyTorch's default) — the cell halves every step at the start.

**13. Fused `nn.LSTM` for training, `ScratchLSTM` for understanding, a test tying them together.**
Why: the hand-written loop is the clearest explanation and the fused kernel is many times faster;
sharing parameter names makes checkpoints interchangeable, and the equivalence test means the
explanation provably describes what was trained.

**14. Random windows with a zero starting state, 256 tokens.** Why: exactly how M3's GPT will be
trained, so the comparison changes one thing — the architecture. Not chosen: *truncated
backpropagation through time with the state carried between consecutive chunks* — the classic
LSTM recipe and probably a little better for the LSTM, but it gives it effectively unlimited
context, which the GPT doesn't get.

**15. AdamW at a peak of 1e-2, β₂ = 0.99, weight decay 0.01 on matrices only.** Why: Adam scales
each parameter's step by its own gradient history, which suits embeddings whose rows get gradients
only when their token appears; AdamW applies weight decay directly instead of through the gradient,
where Adam's scaling would weaken it (Loshchilov & Hutter 2019); decaying biases does nothing
useful. β₂ = 0.99 adapts faster than 0.999 in a short run. 1e-2 because it beat 3e-3 at every
evaluation of a 1,200-step test (§5). Not chosen: *SGD with lr 20–30 and clipping* — the classic
LSTM recipe, but it needs much longer training and careful tuning; *the same rate as the GPT* —
each architecture deserves its own best rate, or the comparison is rigged.

**16. Warm-up then cosine decay to 10%.** Why: early Adam statistics are noisy, so big steps at the
start can blow up; decaying the rate at the end lets the model settle into a lower loss. Not
chosen: *a constant rate* — ends higher; *step decay* — one more thing to tune.

**17. Gradient-norm clipping at 1.0.** Why: recurrent networks can produce rare huge gradients;
rescaling the whole gradient vector keeps its direction but caps the step. Not chosen: *clipping
each value* — changes the direction; *no clipping* — one bad batch can undo hours of training.

**18. Dropout 0.1.** Why: 1.8 epochs is short enough that heavy regularisation would only slow
learning, but the second pass repeats data. Not chosen: *AWD-LSTM's variational and weight dropout*
— stronger, but more machinery than a baseline needs; *0* — risks memorising in epoch 2.

**19. Exactly-resumable checkpoints, atomic writes, `weights_only` loading.** Why: laptops sleep and
runs get interrupted; resuming should not change the result, saving should never corrupt the
previous file, and loading a checkpoint should never execute code. Not chosen: *saving only final
weights* — an interruption loses the run.

**20. Results per language.** Why: Hindi Wikipedia (formal, 8.3 bytes/token) and Hinglish (short,
informal, 3.5 bytes/token) behave very differently; one overall number would hide it.

**21. fp32, no mixed precision.** Why: the model is small, fp32 is the numerically safe default on
Apple GPUs, and one fewer variable while building the pipeline. Revisit in M3 for the bigger GPT.

**22. Train on the Mac's GPU — measured first.** `bench.py` timed both: 39k tokens/s on the M2's GPU
(MPS) vs 15k on its CPU, i.e. 14 vs 37 minutes. Not chosen: *the CPU* — 2.6× slower; *Colab* —
uploading data and session limits buy nothing at this size.

**23. Data quirks left in place for now.** Hinglish-TOP arrives pre-tokenised ("I ' m", " ?") with
some literal "\n" sequences. Cleaning it now would change the data under every M1 and M2 number;
if M5's Hinglish fine-tuning needs clean text, fix it then and re-run the baselines (17 minutes).

**24. Compare per source, and try mixing.** Why: an overall number said "the LSTM wins"; per source
it showed the opposite on Hinglish and explained why (memorisation vs generalisation), and the
mixture turned that into a stronger bar for M3. Not chosen: *reporting only the overall number* —
it hid the most interesting result of the milestone.

## 8. Interview questions you should be able to answer

1. **What is perplexity?** e^(cross-entropy per token): the effective number of equally likely
   choices. It depends on the tokenizer, which is why we report bits per byte.
2. **What should the loss be at initialisation?** ln V (9.01 for V = 8,192) — a near-uniform output.
   Anything much higher means badly scaled logits.
3. **Why do n-gram models need smoothing? Add-k vs interpolation vs Kneser-Ney?** Most long contexts
   are unseen, so raw counts give probability 0. Add-k adds fake counts (bad with big vocabularies);
   interpolation mixes in shorter contexts; Kneser-Ney also uses how many *different* contexts a
   word follows.
4. **Why do plain RNNs struggle with long-range dependencies?** Backprop through time multiplies
   many Jacobians, so gradients vanish or explode.
5. **How does the LSTM fix vanishing gradients?** The cell is updated additively; along it the
   gradient is multiplied by the forget gate, ≈ 1 when remembering.
6. **What is weight tying and why does it help?** Output layer = transposed input embedding: fewer
   parameters, better perplexity, and both roles learn from every token.
7. **Why are LSTMs slow to train compared with Transformers?** Step t needs h_{t−1}, so the 256
   steps run one after another; attention processes all positions of a window in parallel.
8. **What is teacher forcing?** In training, the model always sees the true previous tokens; when
   generating it sees its own samples, so early mistakes compound (exposure bias).
9. **Adam vs AdamW?** AdamW applies weight decay directly to the weights instead of adding it to the
   gradient, where Adam's per-parameter scaling would distort it.
10. **Why warm-up, why cosine decay, why clipping?** Noisy early statistics; settling at the end;
    rare exploding gradients.
11. **What do you need to save to resume training exactly?** Weights, optimiser state, step,
    learning-rate schedule position, and every random-number generator's state.
12. **Why is language modelling the same as compression?** Arithmetic coding turns a model that
    assigns probability p to the text into a code of −log₂ p bits; better prediction = smaller file.
13. **How does generation cost differ between an LSTM and a GPT?** LSTM: constant work per token
    from a fixed-size state. GPT: attends to the whole context each step (a KV-cache avoids
    recomputing it).
14. **Your LSTM beat the 4-gram overall but lost on Hinglish. Why?** Validation Hinglish mostly
    repeats training phrases (only 2.4% of Hinglish-TOP's 4-grams are new), which exact counts
    replay perfectly; the small network, trained on 77% Hindi, spends its capacity on Hindi, where
    the text is new and generalisation wins.
15. **Why does mixing two language models help?** They make different mistakes. And since −log is
    convex, −log(w·p + (1−w)·q) ≤ w·(−log p) + (1−w)·(−log q): the mixture's loss is never worse
    than the weighted average of the two losses, and better wherever they disagree.

## 9. Try it yourself

- Implement Kneser-Ney smoothing for the bigram and trigram; how much does it beat JM?
- `python scripts/train.py --run scratch --scratch true --max-iters 20 --eval-interval 20`: how many
  times slower is the hand-written LSTM?
- Train with `--forget-bias 0`, `--dropout 0` or `--dropout 0.3`; compare the curves with
  `python scripts/report.py`.
- Reproduce the initialisation plateau: `--init-range 0.035` (≈ the spread of N(0, 0.02)).
- Change `--temperature` in `scripts/sample.py` to 0, 0.5, 1.0, 1.5 and describe what happens.
- Carry the LSTM state from one window to the next during evaluation; how much does it gain?
- Train the LSTM for 8,000 steps (`--max-iters 8000`): how much of the gap to the mixture closes?
