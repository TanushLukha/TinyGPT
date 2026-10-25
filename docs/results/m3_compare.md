# Where each model wins

Bits per byte on the validation set, by data source (lower is better). The unseen column is the share of the source's 4-grams (a token and the 3 before it) that never occur in the training data: low means the text repeats training phrases.

| source | tokens | bytes/token | 4-grams unseen in training | 4-gram | gpt-llama | gpt | gpt-small | lstm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki_hi | 79,649 | 8.27 | 68.6% | 0.829 | 0.689 | 0.735 | 0.770 | 0.785 |
| findnitai | 12,332 | 3.32 | 24.6% | 1.233 | 1.303 | 1.405 | 1.488 | 1.435 |
| hinglish_top | 7,872 | 3.73 | 2.4% | 0.746 | 0.931 | 1.012 | 1.083 | 1.029 |

## Mixing gpt-llama with the 4-gram

p = w · p_gpt-llama + (1 − w) · p_4-gram, for every token; the weight w is chosen on validation.

| weight of gpt-llama | Hindi | Hinglish | all |
|---:|---:|---:|---:|
| 0.0 | 0.829 | 1.029 | 0.848 |
| 0.1 | 0.751 | 0.984 | 0.774 |
| 0.2 | 0.724 | 0.968 | 0.748 |
| 0.3 | 0.706 | 0.961 | 0.731 |
| 0.4 | 0.693 | 0.958 | 0.719 |
| 0.5 | 0.683 | 0.959 | 0.710 |
| 0.6 | 0.676 | 0.965 | 0.704 |
| 0.7 | 0.671 | 0.977 | 0.701 |
| **0.8** | 0.669 | 0.995 | 0.701 |
| 0.9 | 0.671 | 1.029 | 0.705 |
| 1.0 | 0.689 | 1.147 | 0.733 |

## Loss vs context (wiki_hi)

Mean loss (nats/token) by how many tokens of its 256-token window a model had read when predicting. The 4-gram always looks 3 tokens back, so it barely changes.

| tokens seen | 4-gram | gpt-llama | gpt | gpt-small | lstm |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.688 | 5.678 | 5.451 | 5.393 | 5.481 |
| 2 | 4.658 | 4.775 | 4.865 | 4.910 | 4.865 |
| 3–4 | 4.690 | 4.704 | 4.805 | 4.945 | 4.827 |
| 5–8 | 4.704 | 4.449 | 4.583 | 4.718 | 4.657 |
| 9–16 | 4.773 | 4.246 | 4.434 | 4.587 | 4.594 |
| 17–32 | 4.683 | 3.988 | 4.218 | 4.404 | 4.454 |
| 33–64 | 4.727 | 3.914 | 4.139 | 4.325 | 4.418 |
| 65–128 | 4.753 | 3.937 | 4.196 | 4.398 | 4.492 |
| 129–256 | 4.764 | 3.890 | 4.187 | 4.404 | 4.501 |

![loss vs context](m3_context.png)

## Copying a random sequence

Loss (nats/token) on 32 sequences of 100 random tokens, each shown twice. The first copy can only be guessed; the second can be copied from the first by a model that can look back that far.

| model | 1st copy | 2nd copy |
|---|---:|---:|
| gpt-llama | 13.19 | 11.90 |
| gpt | 13.26 | 12.97 |
| gpt-small | 13.45 | 13.32 |
| lstm | 13.27 | 13.16 |
