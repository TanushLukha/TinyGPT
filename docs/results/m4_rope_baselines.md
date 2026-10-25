# M4_ROPE — results

Validation set: every token of `data/tokens/val.bin` (99.9k tokens). Bits per byte = total cross-entropy in bits ÷ UTF-8 bytes of the text, so it doesn't depend on the tokenizer. Lower is better.

| model | size | loss/token | bits/byte | Hindi | Hinglish | training |
|---|---:|---:|---:|---:|---:|---|
| uniform | — | 9.011 | 1.781 | 1.572 | 3.738 | — |
| unigram | 8.1k n-grams | 7.499 | 1.482 | 1.303 | 3.166 | counting |
| bigram | 1.7M n-grams | 4.973 | 0.983 | 0.915 | 1.617 | counting |
| trigram | 8.9M n-grams | 4.437 | 0.877 | 0.838 | 1.239 | counting |
| 4-gram | 20.6M n-grams | 4.291 | 0.848 | 0.829 | 1.029 | counting |
| **gpt** (gpt) | 13.9M params | 3.965 | **0.784** | 0.735 | 1.241 | 32.8M tokens, 50 min (mps) |
| **gpt-rope** (gpt) | 13.8M params | 3.730 | **0.737** | 0.691 | 1.168 | 32.8M tokens, 46 min (mps) |
| **gpt-llama** (gpt) | 13.8M params | 3.708 | **0.733** | 0.689 | 1.147 | 32.8M tokens, 53 min (mps) |

![learning curves](m4_rope_curves.png)
