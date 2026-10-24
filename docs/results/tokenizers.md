# Tokenizer comparison

Validation text — Hindi: 0.66 MB, Hinglish: 0.07 MB. Lower tokens/word is better.

| Tokenizer | Vocab | Hindi tok/word | Hindi bytes/tok | Hindi 1-tok % | Hinglish tok/word | Hinglish bytes/tok | Hinglish 1-tok % |
|---|---|---:|---:|---:|---:|---:|---:|
| TinyGPT BPE (ours) | 8,192 tokens | 1.65 | 8.28 | 65% | 1.47 | 3.49 | 77% |
| same BPE, GPT-4 regex | 8,192 tokens | 2.67 | 5.12 | 10% | 1.34 | 3.85 | 83% |
| SentencePiece BPE | 8,192 tokens, same sample | 1.67 | 8.19 | 66% | 1.52 | 3.38 | 82% |
| SentencePiece UNIGRAM | 8,192 tokens, same sample | 1.64 | 8.31 | 69% | 1.53 | 3.37 | 85% |
| GPT-2 | 50,257 tokens | 8.11 | 1.68 | 2% | 1.69 | 3.05 | 57% |
| GPT-4 (cl100k) | 100,277 tokens | 5.28 | 2.58 | 1% | 1.58 | 3.26 | 60% |
| GPT-4o (o200k) | 200,019 tokens | 1.83 | 7.46 | 57% | 1.46 | 3.52 | 69% |
| characters | Unicode code points | 5.36 | 2.55 | 1% | 5.14 | 1.00 | 6% |
| UTF-8 bytes | Devanagari = 3 bytes/char | 13.64 | 1.00 | 1% | 5.14 | 1.00 | 6% |
