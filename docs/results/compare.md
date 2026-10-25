# Where each model wins

Bits per byte on the validation set, by data source (lower is better). The unseen column is the share of the source's 4-grams (a token and the 3 before it) that never occur in the training data: low means the text repeats training phrases.

| source | tokens | bytes/token | 4-grams unseen in training | 4-gram | lstm |
|---|---:|---:|---:|---:|---:|
| wiki_hi | 79,649 | 8.27 | 68.6% | 0.829 | 0.785 |
| findnitai | 12,332 | 3.32 | 24.6% | 1.233 | 1.435 |
| hinglish_top | 7,872 | 3.73 | 2.4% | 0.746 | 1.029 |

## Mixing lstm with the 4-gram

p = w · p_lstm + (1 − w) · p_4-gram, for every token; the weight w is chosen on validation.

| weight of lstm | Hindi | Hinglish | all |
|---:|---:|---:|---:|
| 0.0 | 0.829 | 1.029 | 0.848 |
| 0.1 | 0.783 | 0.996 | 0.804 |
| 0.2 | 0.766 | 0.989 | 0.787 |
| 0.3 | 0.754 | 0.989 | 0.777 |
| 0.4 | 0.747 | 0.994 | 0.771 |
| 0.5 | 0.742 | 1.002 | 0.767 |
| **0.6** | 0.740 | 1.016 | 0.766 |
| 0.7 | 0.740 | 1.035 | 0.768 |
| 0.8 | 0.743 | 1.064 | 0.774 |
| 0.9 | 0.751 | 1.111 | 0.786 |
| 1.0 | 0.785 | 1.265 | 0.831 |
