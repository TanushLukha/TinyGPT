---
title: TinyGPT
emoji: 🪔
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
license: mit
short_description: A Hindi & Hinglish GPT that runs in your browser
models:
  - {repo}
---

# TinyGPT — Hindi & Hinglish, in your browser

A 13.8M-parameter GPT trained from scratch on Hindi Wikipedia and Hinglish, and a twin of it
fine-tuned to parse Hinglish requests to a voice assistant.

This Space is a static page: no server runs the model. The visitor's browser downloads the weights
from [{repo}](https://huggingface.co/{repo}) (commit `{revision}`), checks their SHA-256 and runs
them with a JavaScript port of the model — tokenizer, forward pass and sampling — checked against
the PyTorch code it was ported from. What you type never leaves your browser.

* **Code and write-ups:** https://github.com/TanushLukha/TinyGPT
* **Model weights:** https://huggingface.co/{repo}
