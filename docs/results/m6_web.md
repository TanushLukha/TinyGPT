# M6 — the browser version against PyTorch

`scripts/check_web.py`: `web/tinygpt.js` (the model rewritten in JavaScript, which the demo page runs) against the Python/PyTorch code it was ported from, with the real bundles. Linux x86_64, Node.js v22.23.4, PyTorch 2.14.1+cu130.

| check | result |
|---|---|
| tokenizer: texts encoded to the same token ids | 5,503 of 5,503 (113,053 tokens) |
| language model: largest logit difference, 10 prompts | 3.8e-06 |
| language model: identical 60-token greedy continuations | 10 of 10 |
| parser: identical trees, first 200 validation requests | 200 of 200 |
| parser: exact match, browser / server | 49.0% / 49.0% |
| speed in Node.js (one thread): time per token | 35.1 ms |
