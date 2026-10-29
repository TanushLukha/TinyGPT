# M5 Fine tuning: LoRA from scratch, on a Hinglish parsing task

**Goal:** teach the pretrained TinyGPT a real task with very little labelled data, and understand
the options: training from scratch, full fine tuning, few shot prompting, and **LoRA** (written
from scratch), which trains about 2% of the weights. Along the way: how to turn a structured task
into language modelling, how to evaluate it honestly, and what to do when the test set turns out
to be inside the pretraining data.

**In short:** on 6,513 test requests, full fine tuning gets 38.0% of parses exactly right (41.0% with
3× the training), LoRA 19.5% while training 2.3% of the weights (32.4% with 3× the training), the
same network trained from scratch 19.1%, and few shot prompting 0%. At 14M parameters LoRA mostly
learns *slower*; its rank and the MLP adapters matter; and the test requests hidden in the
pretraining data left a trace in the language model but didn't help parsing.

```
request ─┐                                             ┌─ [in:create_alarm [sl:date_time kal subah 7 baje ] ]
         └─ <|endoftext|> kal subah 7 baje ka alarm laga do => ─ TinyGPT (frozen) + LoRA (2% of the weights) ─┘
```

## 1. The task: Hinglish TOP

Voice assistants turn a request into an *intent* (what to do) and *slots* (the details). Hinglish TOP
(Agarwal et al., 2022; Apache 2.0) has 10,896 human written Hinglish requests from 8 domains (alarm,
event, messaging, music, navigation, reminder, timer, weather), each with a parse tree:

```
do ghante ke liye alarm set kardo
[IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] ]

agar mai raat 9 baje tak idhar se nikal jau toh kya mai raat 11 baje tak Taipei pahunch jaunga ?
[IN:GET_ESTIMATED_ARRIVAL [SL:DATE_TIME_DEPARTURE raat 9 baje tak ]
   [SL:SOURCE [IN:GET_LOCATION [SL:LOCATION_USER idhar ] ] ] [SL:DATE_TIME_ARRIVAL raat 11 baje tak ] [SL:DESTINATION Taipei ] ]
```

| | training | validation | test |
|---|---:|---:|---:|
| requests | 2,993 | 1,390 | 6,513 |
| words per request | 8.8 | 8.6 | 8.8 |
| parses with a nested intent | 12.7% | 13.6% | 15.3% |

75 intents and 69 slot types in all; only 26 test requests (0.4%) use a label never seen in training.
The training set is small on purpose: this is the regime where pretraining and LoRA matter.

**What is scored** is the *decoupled* parse (Aghajanyan et al., 2020): only the words inside slots are
kept (what an assistant needs to act on), so `alarm set kardo` disappears from the first example
above. **What the model writes** can be that decoupled tree, or the whole request *annotated* with
brackets (the full tree, decoupled before scoring): §5 explains why this matters for a small model,
and the choice is made on validation data. **Scores**, all on the decoupled tree:
- **exact match**: the whole tree is right (the standard "frame accuracy");
- **intent accuracy**: the top intent is right;
- **slot F1**: over (slot label, slot text) pairs;
- **valid trees**: the output parses as a tree at all.

Two reference points that need no learning: always answering the most common intent
(`[IN:GET_WEATHER ]`) gets 13.9% intent accuracy and 0.8% exact match; a lookup table that copies the
parse of an identical training request gets 9.3% exact match: 639 test requests (9.8%) repeat a
training request, so they are also scored separately.

## 2. Found before training: the test set was in the pretraining data

M1 put every Hinglish TOP request into the pretraining corpus, including the validation and test
requests (not their parses). Checking before any fine tuning: **99.5% of the test requests**
(6,481 of 6,513) were in the pretraining text. A model that has read the test inputs may parse them
better than truly new requests: a mild form of *test contamination*.

The fix (`scripts/decontaminate.py`): remove every pretraining line that matches a validation or test
request, ignoring case, punctuation and spacing (8,095 of 140,522 Hinglish TOP lines, 0.55% of all
training tokens), and pretrain **`gpt-llama-clean`** exactly like M3's `gpt-llama` on the rest.
Training requests stay (fine tuning reads them anyway); the validation text for bits per byte stays,
so the clean model's score is comparable with M3's. Then the effect can be *measured*: fine tune both
base models the same way and compare (§6). The tokenizer was also trained on text containing these
requests; it only learns frequent character sequences, so it stays.

## 3. Four ways to teach a model a task

| | what trains | idea |
|---|---|---|
| **few shot prompting** | nothing | put a few solved examples in the prompt and let the model continue the pattern |
| **from scratch** | all 13.8M weights, from random | the same network, but without pretraining |
| **full fine tuning** | all 13.8M weights, from the pretrained ones | continue training on the task |
| **LoRA** | 0.3M new weights; the 13.8M stay frozen | learn a small *change* to each weight matrix |

## 4. LoRA from scratch (`src/tinygpt/lora.py`)

Fine tuning changes a weight matrix W (out × in) to W + ΔW. LoRA (Hu et al., 2021) bets that ΔW
doesn't need all out × in degrees of freedom (that the update has a low "intrinsic rank") and
writes it as a product of two thin matrices:

```
y = x Wᵀ + b + (α/r) · x Aᵀ Bᵀ        A: r × in (trainable)   B: out × r (trainable)   W, b: frozen
```

- **Few numbers.** Our attention projection (1,152 × 384) has 442,368 weights; with r = 8, A and B
  hold 8 × (384 + 1,152) = 12,288. Over all 30 linear layers of the six blocks:
  **313,344 trainable weights, 2.3% of the model**, a 1.3 MB adapter file instead of 55 MB.
- **Starts exactly at the pretrained model.** B = 0, so ΔW = 0 at step 0 (`test_starts_exactly_as_the_pretrained_model`).
  A starts random (Kaiming uniform, as in Microsoft's reference code `loralib`; the paper says
  Gaussian): if both were zero, neither would ever get a gradient: ∂L/∂B needs Ax ≠ 0 and ∂L/∂A
  needs B ≠ 0.
- **α/r.** The update is scaled by α/r with α fixed (16): the paper sets α to the first r it tries and
  never tunes it, so changing r needs no new learning rate.
- **Where.** The paper adapted only the attention's W_q and W_v; QLoRA (Dettmers et al., 2023) found
  that "LoRA on all linear transformer block layers" is required to match full fine tuning. We adapt
  all of them (`targets=("attn", "mlp")`) and measure attention only too. Embeddings and the output
  layer stay frozen: the vocabulary doesn't change.
- **Merging.** After training, W ← W + (α/r)·BA and the wrapper is removed (`merge_lora`): an ordinary
  GPT again, with no extra cost per token (`test_merging_gives_the_same_model`). Or keep the adapter
  separate: one frozen base, many 1.3 MB adapters, one per task.

**What LoRA saves, and what it doesn't.** Memory: gradients and Adam's two running averages exist
only for the 0.3M trainable weights (Adam alone: 2.5 MB instead of 110 MB here): at GPT 3's size the
paper reports 10,000× fewer trainable parameters and 3× less GPU memory. Compute: the forward and
backward passes still go through every layer: only the weight gradient products of the frozen
matrices are skipped. And forgetting: *while the adapter is on*, the model's behaviour on other text
changes too; the difference is that removing the adapter restores the base model exactly.

## 5. The task as language modelling (`src/tinygpt/top.py`, `scripts/finetune.py`)

Every example becomes one sequence:

```
<|endoftext|>do ghante ke liye alarm set kardo => [in:create_alarm [sl:date_time do ghante ke liye ] ]<|endoftext|>
└──────────────── prompt: no loss ─────────────┘ └──────────────────── loss ──────────────────────────────┘
```

- **Loss only on the parse.** Targets inside the prompt are set to minus 100, which cross entropy ignores:
  the model learns p(parse | request), not to imitate requests. It also learns to end with
  <|endoftext|>, so generation knows when to stop.
- **Decoupled or annotated output.** With the decoupled tree, the model must *pick* the slot words out
  of the request: after `[sl:location` it has to find `hawaii` somewhere in the prompt. In the first
  runs it often didn't: it invented plausible words instead (`[sl:weather_attribute half hour ]` for
  "kya hawaii me raining ho rahi hai?"). The annotated form makes it *copy the request in order*,
  inserting brackets: the next word is always the word after the last one copied, exactly what the
  induction heads of M3 do. It is only 16% longer (43.8 vs 37.9 tokens per parse). Which works better
  is decided on validation: LoRA r = 8 at the same learning rate, once per format; the better one is
  used for every other run.
- **Labels in lower case.** Our tokenizer never saw `[IN:CREATE_ALARM` and spells it letter by letter
  (14 tokens); `[in:create_alarm` is 8, and contains pieces the model knows (`alarm`). Parses get 31%
  shorter (36.9 instead of 53.7 tokens on average). The mapping is exact both ways, so scores are
  computed on the dataset's own format; §6 measures what it changes.
- **Batches of similar lengths.** Examples run from 15 to 200 tokens (median 47); in random batches
  57% of the compute would go to padding. Sorting within chunks of 50 batches cuts that to 3%; every
  example is still seen once per epoch, in a new order.
- **Same budget for every method:** 600 steps of 32 examples (6.4 epochs of the training set), warmup
  then cosine decay, AdamW, gradient clipping at 1.0. The learning rate is chosen per method on
  validation exact match: LoRA {0.001, 0.003, 0.01}, full fine tuning {0.0001, 0.0003, 0.001}, from scratch
  {0.0003, 0.001, 0.003} (the first runs also tried 0.0003 for LoRA and 0.00003 for full fine tuning: far
  worse). Most runs were still improving at step 600, so the two main methods are also trained 3×
  longer, to see whether the gap between them closes.
- **Examples that don't fit are skipped.** The model reads at most 256 tokens. With upper case labels,
  2 training parses are longer than that (they crashed the first run); they are left out of training
  and the count is recorded.
- **Checkpoint by exact match.** Every 150 steps the model parses all 1,390 validation requests
  (greedy decoding, M4's KV cache, until <|endoftext|> or the context window is full) and the best
  weights are kept. Requests of equal length are decoded together, so no padding is needed and each
  is read exactly as in training.
- **The test set is used once per configuration,** after the format and the learning rate are fixed.

## 6. Results

All numbers on the 6,513 test requests unless marked; the base model is `gpt-llama-clean`; the model
writes the annotated request (chosen on validation, below). Full tables: `docs/results/m5_finetune.md`.

| method | trainable weights | to store per task | test exact match | intent | slot F1 | general text after (bits/byte) |
|---|---:|---:|---:|---:|---:|---:|
| lookup table (copy the parse of an identical training request) | 0 | n/a | 9.3% | 9.7% | 11.0% | n/a |
| few shot: 3 solved examples in the prompt | 0 | n/a | 0.0% | 0.0% | 0.0% | 0.731 |
| from scratch | 13.8M | 55 MB | 19.1% | 80.5% | 28.1% | 2.001 |
| **full fine tuning** | 13.8M | 55 MB | **38.0%** | 83.2% | **53.4%** | 1.519 |
| LoRA r = 8, attention + MLP | 313K | 1.3 MB | 19.5% | 77.5% | 29.9% | 0.873 |
| LoRA r = 8, attention only | 111K | 0.4 MB | 5.1% | 65.4% | 13.7% | 0.783 |
| *3× longer training:* full fine tuning | 13.8M | 55 MB | **41.0%** | 83.9% | 56.8% | 2.069 |
| *3× longer training:* LoRA r = 8 | 313K | 1.3 MB | 32.4% | 82.1% | 46.7% | 1.051 |

![exact match vs trainable weights](results/m5_rank.png)

**What we learned**

- **At this size, full fine tuning wins, and LoRA mostly learns slower.** With the same 600 steps,
  LoRA reaches half of full fine tuning (19.5% vs 38.0%). Given 3× the steps it climbs to 32.4%, while
  full fine tuning gains only 3 points (41.0%): most of the gap was speed, not capacity. Both were
  still improving at step 1,800.
- **Rank matters here.** r = 1 gets 3.1%, r = 4 13.0%, r = 8 19.5%, r = 16 22.4%, r = 64 25.9%; the
  "rank 1 to 2 is enough" result of the LoRA paper was for GPT 3, a model more than 10,000× larger, with
  far more redundancy to exploit. And the MLP layers matter most: attention only adapters get 5.1%.
- **Pretraining is worth a 2× better parser.** Full fine tuning beats the same network trained from
  scratch at every data size (23.7% vs 14.7% with 1,000 examples, 38.0% vs 19.1% with all of them).
  With 100 examples nothing works (2 to 3%).

![exact match vs number of training examples](results/m5_data.png)

- **The base model cannot learn from examples in its prompt.** After "=>" its first token is a line
  break: it treats the demonstrations as text to continue, not a pattern to follow. In context learning
  needs far bigger models.
- **Seeds:** three LoRA runs land at 19.5%, 19.5% and 19.8%: differences under about half a point are
  noise.
- **Forgetting.** On general text, full fine tuning goes from 0.731 to 1.519 bits per byte (2.069 after
  1,800 steps); the LoRA model, adapter on, to 0.873, and the base comes back exactly when the adapter
  is removed. One base model can carry many such adapters.

**Did reading the test requests during pretraining help?** No, though it did leave a trace:

| base model | saw the test requests | LoRA | full fine tuning | base loss on test requests | on training requests (control) |
|---|---|---:|---:|---:|---:|
| `gpt-llama` (M3) | yes (not their parses) | 19.2% | 37.5% | 3.222 nats/token | 2.669 |
| `gpt-llama-clean` | no | 19.5% | 38.0% | 3.457 | 2.759 |

The contaminated model finds the test requests 0.235 nats/token easier to predict, against 0.090 on
requests both models read: it had partly memorised them. But its parsers are no better (0.3 and 0.5
points *lower*, about the size of the seed to seed spread): knowing a request's wording doesn't tell
you its parse.

**What the model writes** (validation, LoRA r = 8 at the same learning rate): the annotated request
28.4% vs the decoupled tree 27.7%: a small win, so the annotated form was used. On test, at each
method's best rate: LoRA 19.5% vs 19.1%, from scratch 19.1% vs 18.7%, full fine tuning 38.0% vs 32.6%
(the latter partly a different learning rate). Copying in order helped less than hoped: the model
copies common words well but often swaps rarer ones (above all names it never saw in training) for
ones it did see:

```
mujhe Delhi ka weather batao          → [in:get_weather mujhe [sl:location Duran ] ka weather batao ]
Arijit Singh ke gaane bajao           → [in:play_music [sl:music_artist_name Arizar ] [sl:music_type album ] bajao ]
```

(The training requests come from a US assistant: New York appears 28 times, Chicago 9, Taylor Swift 5;
Delhi and Arijit Singh never.)

**Lower case labels** beat the dataset's upper case: 19.5% vs 16.4%, and 12% less training time (the
upper case form is 59.6 tokens per parse against 42.8; 4 training examples didn't even fit the window).

**Where it fails** (LoRA, test errors): a slot missing, extra or mislabelled 41%; wrong intent 24%;
right slots but wrong words in one 22% (`raining` → `cutesting`); not a valid tree 13%. Requests that
also appear in training are parsed far better (LoRA 67.1% vs 14.3% on new ones, full fine tuning 94.5%
vs 31.9%), and 26% of *validation* requests repeat a training request against 9.8% of test ones,
which is why validation scores run 9 to 11 points higher. By domain, formulaic requests are easiest
(weather 57.1%, alarm 55.9%, timer 48.6% for full fine tuning) and free text slots hardest (reminder
16.6%, navigation 22.8%, messaging 25.9%).

**Limits of these numbers.** Full fine tuning's best learning rate (0.001) is the top of its grid; a
higher one might do better still. Every main run was still improving at its last step. Each base
model was pretrained once.

Try the best parser: `python scripts/parse.py --out-dir runs/m5-annotated --run full-long "kal subah 7 baje ka alarm laga do"`.

## 7. Decisions we made, and why

Each entry: what we chose → why → the alternatives and why we didn't pick them.

**1. Task: Hinglish TOP semantic parsing.** Why: a real assistant task in our languages, a permissive
licence, human written data, an unambiguous score (the tree is right or not), and a small training
set: the regime fine tuning methods differ in. Not chosen: *English → Hinglish translation*
(findnitai): our model has read little English, and BLEU style scores are hard to interpret;
*sentiment classification*: a 3 way label tests less of what a language model learned.

**2. Retrain a clean base model rather than just reporting the contamination.** Why: 99.5% of the
test requests were in the pretraining text; a clean model makes the main numbers trustworthy *and*
lets us measure what the contamination was worth (answer: nothing for parsing, see §6). Cost: one more
50 minute pretraining run. Not
chosen: *a footnote*: honest, but leaves the size of the effect unknown; *n gram overlap filtering
(remove any line sharing a long n gram with a test request)*: requests are 9 words long, so whole line
matching is the natural unit.

**3. Score the decoupled parse; let validation decide what the model writes.** Why: the decoupled tree
holds everything an assistant acts on, and exact match on it is the standard metric, so it is what we
score, whatever the model writes. *Writing* it means picking words out of the request, which our
first runs did badly (they invented slot words); writing the annotated request turns that into
copying in order. Which is better for a 14M model is an empirical question, so one LoRA run per
format decides it on validation: the annotated form won narrowly (28.4% vs 27.7%). Not chosen:
*fixing the format by argument alone*: neither argument (shorter output vs easier copying) predicts
by how much, and here both turned out to matter little.

**4. Lower case labels.** Why: 28 to 31% fewer tokens per parse, and label pieces the model has seen as
words. The mapping is exact, so scores are unaffected by the choice itself. Measured: 19.5% exact match
vs 16.4% with the dataset's upper case. Not chosen: *new tokens,
one per label*: shortest of all, but the embedding table must grow and the new rows must be trained,
which breaks "only the adapter trains".

**5. Loss on the parse only.** Why: the request is given at test time; spending capacity on modelling
it is wasted, and it would let a model lower its loss without parsing better. Not chosen: *loss on the
whole sequence*: simpler, an exercise below.

**6. LoRA written as a wrapper module.** `LoRALinear` holds the original nn.Linear and adds A and B.
Why: works on any linear layer without touching the GPT code, and merging just unwraps it. Not chosen:
*a subclass of nn.Linear* (loralib's way): needs its own constructor for every layer;
*the PEFT library*: the point of M5 is to build it.

**7. α = 16 for every rank (scale α/r).** Why: the paper's rule; the rank sweep doesn't need a new
learning rate for each r. Not chosen: *α = 2r* (constant scale): would need retuning; *α/√r*
(rsLoRA): an exercise.

**8. Adapt every linear layer of every block, r = 8.** Why: QLoRA's finding; r = 8 is a common default
and the rank sweep shows how much it matters. Not chosen: *only W_q and W_v* (the original paper):
measured as "attention only": 5.1% against 19.5%.

**9. Explicit merging.** `merge_lora` folds and unwraps once. Not chosen: *merging automatically on
`model.eval()` and unmerging on `model.train()`* (loralib): convenient, but it silently edits
weights on every mode switch, and repeated add/subtract drifts in floating point.

**10. Same steps, batch and schedule for every method; learning rate chosen per method.** Why: a
method shouldn't win because it got a better learning rate. Not chosen: *one learning rate for all*:
LoRA's best rate is typically far higher than full fine tuning's, so a shared rate favours one of them.
When a best rate sat at the edge of its grid (full fine tuning at 0.0003 in the first runs), the grid
was extended past it, and full fine tuning's best (0.001) is again its top value, a limit we report.

**10b. A fixed budget of 600 steps, plus one 3× longer run per main method.** Why: most runs were still
improving at step 600, so a fixed budget compares how fast methods learn as much as how far they can
go; the 1,800 step runs show whether LoRA's gap to full fine tuning is speed or capacity. Mostly
speed: LoRA went from 19.5% to 32.4%, full fine tuning from 38.0% to 41.0%. Not chosen: *training
every run 3× longer*: triples a 3 hour experiment for an answer two runs can give.

**11. Choose checkpoints by validation exact match.** Why: it is the score we care about; validation
loss can keep improving while exact match doesn't, and vice versa. Cost: greedy decoding of 1,390
requests every 150 steps.

**12. No weight decay.** Why: runs are short; and for full fine tuning, decay pulls weights towards
zero, away from the pretrained values, not towards them. Same setting for all methods.

**13. Greedy decoding.** Why: deterministic, and a parse has one right answer; M4's samplers trade
accuracy for variety, the wrong trade here. Not chosen: *beam search*, *constrained decoding* (only
valid labels and brackets): both exercises.

**14. Group requests by length for decoding, not padding.** Why: equal length prompts need no padding,
so each is read exactly as in training, with no change to the model. Not chosen: *left padding*:
fewer, larger batches, but needs an attention mask for the pads in every layer.

**15. Batches of similar lengths for training.** Why: padding drops from 57% to 3% of the compute.
Not chosen: *random batches*: simpler, more than half the compute wasted.

**16. Report requests seen in training separately; three seeds for the main run.** Why: 9.8% of test
requests repeat a training request, and a lookup table already gets 9.3%; seeds tell us which
differences are real (the three LoRA runs: 19.5%, 19.5%, 19.8%).

**17. Few shot with three fixed random examples.** Why: the simplest in context baseline. Not chosen:
*retrieving the most similar training examples for each request*: a stronger baseline, an exercise.
Its answer is the first nonempty line after "=>": the base model's first token there is a line break,
so cutting at the first line would score an empty answer for everything.

**18. Skip training examples longer than the context window.** Why: a 256 token model cannot read them;
truncating would teach it parses with the end cut off. Only the upper case label runs have any (2 of
2,993); the count is saved in each run's results.

## 8. Interview questions you should be able to answer

1. **What is LoRA?** Freeze W; learn ΔW = (α/r)·BA with B (out × r) and A (r × in), r ≪ in, out. Only
   A and B train; they can be merged into W afterwards.
2. **Why does a low rank update work?** The change fine tuning needs has a low "intrinsic rank"
   (Hu et al., 2021): for GPT 3, rank 1 to 2 was enough on some tasks. On our 14M model it was not:
   r = 1 got 3.1%, r = 64 25.9%: a small model has little redundancy to exploit.
3. **Why is B initialised to zero and A randomly?** B = 0 makes the model start exactly at the
   pretrained one; A ≠ 0 lets B get a gradient from the first step (∂L/∂B ∝ Ax). Both zero: no learning.
4. **What does α/r do?** Scales the update so that changing r doesn't change its size much: the
   learning rate can stay. α is set once and not tuned.
5. **Where do LoRA's memory savings come from?** No gradients or optimizer states for frozen weights
   (Adam keeps two numbers per trained weight). Activations are still stored for backpropagation.
6. **Is LoRA faster per step?** Barely: the backward pass skips the weight gradients of frozen
   layers, but forward and backward still run through every layer (600 steps: 2.24 vs 2.31 minutes
   here). Its real savings are memory and storage.
7. **Does LoRA add inference latency?** Not if merged: W + ΔW is one matrix. Unmerged, it costs two
   thin extra products per layer: the price of swapping adapters per request.
8. **Which layers should get adapters?** All linear layers of the blocks match full fine tuning
   (QLoRA); attention only is the original, cheaper setup.
9. **When would you prefer full fine tuning?** Lots of data, a large shift from the pretraining
   domain, or a model small enough that memory doesn't matter, like ours: 38.0% vs LoRA's 19.5% at
   600 steps (41.0% vs 32.4% at 1,800).
10. **What is catastrophic forgetting, and does LoRA prevent it?** Fine tuning degrades what the model
    could do before. With the adapter on, LoRA changes the model too, but removing it restores the
    base exactly, and one base can serve many tasks.
11. **Why mask the prompt in the loss?** So the model learns p(answer | prompt) instead of spending
    capacity on modelling prompts it will always be given.
12. **How do you evaluate a semantic parser?** Exact match of the canonical tree (frame accuracy), plus
    intent accuracy, slot F1 and the share of valid trees, to see *where* it fails.
13. **What is test contamination, and how did you handle it?** Test data inside the training data. We
    found the test requests in the pretraining corpus, removed them, retrained the base, and measured
    the difference.
14. **Why choose hyperparameters on validation and look at test once?** Choosing on test fits the test
    set: the reported number would be optimistic.
15. **What is QLoRA?** LoRA on top of a base model stored in 4 bits: the frozen weights take a quarter of
    the memory, the adapters train in higher precision.
16. **Did the contamination matter?** The contaminated base predicted test requests 0.235 nats/token
    better (0.090 on a control set): it had memorised them, yet parsed them no better. Measuring beat
    guessing either way.
17. **Why was validation exact match 9 to 11 points above test?** 26% of validation requests also appear in
    training, against 9.8% of test ones, and repeated requests are parsed far better (full fine tuning:
    94.5% vs 31.9%). Choosing on validation is still right; quoting it as the result would not be.

## 9. Try it yourself

- Put the loss on the whole sequence instead of the parse only: does exact match change?
- Constrained decoding: allow only labels that exist, brackets that keep the tree valid, and (in the
  annotated form) only the next word of the request. That rules out invented slot words
  (`Delhi` → `Duran`), part of the 22% of LoRA's errors with the right slots but wrong words; how much
  exact match it buys is the experiment.
- Few shot with the 3 most similar training requests (word overlap) instead of random ones.
- rsLoRA: scale by α/√r instead of α/r and redo the rank sweep.
- Train a second adapter for another task on the same base (e.g. English → Hinglish with findnitai)
  and switch adapters per request.
- Quantize the frozen base to 8 bits (QLoRA style) and train the same adapter: how much does exact
  match drop?
