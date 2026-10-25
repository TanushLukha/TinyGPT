# M2 — all baselines

Validation set: every token of `data/tokens/val.bin` (99.9k tokens). Bits per byte = total cross-entropy in bits ÷ UTF-8 bytes of the text, so it doesn't depend on the tokenizer. Lower is better.

| model | size | loss/token | bits/byte | Hindi | Hinglish | training |
|---|---:|---:|---:|---:|---:|---|
| uniform | — | 9.011 | 1.781 | 1.572 | 3.738 | — |
| unigram | 8.1k n-grams | 7.499 | 1.482 | 1.303 | 3.166 | counting |
| bigram | 1.7M n-grams | 4.973 | 0.983 | 0.915 | 1.617 | counting |
| trigram | 8.9M n-grams | 4.437 | 0.877 | 0.838 | 1.239 | counting |
| 4-gram | 20.6M n-grams | 4.291 | 0.848 | 0.829 | 1.029 | counting |
| **lstm** (lstm) | 5.5M params | 4.206 | **0.831** | 0.785 | 1.265 | 32.8M tokens, 16 min (mps) |

![learning curves](m2_curves.png)
