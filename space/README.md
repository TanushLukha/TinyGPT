---
title: TinyGPT
emoji: 🪔
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
license: mit
short_description: A Hindi & Hinglish GPT, built from scratch
models:
  - {repo}
preload_from_hub:
  - {repo} {files} {revision}
---

# TinyGPT — Hindi & Hinglish, built from scratch

A 13.8M-parameter GPT trained from scratch on Hindi Wikipedia and Hinglish, and a twin of it
fine-tuned to parse Hinglish requests to a voice assistant. Everything — tokenizer, model,
training, decoding, LoRA, serving — is hand-written in PyTorch.

* **Code and write-ups:** https://github.com/TanushLukha/TinyGPT
* **Model weights:** https://huggingface.co/{repo}

This Space runs on ZeroGPU: each request borrows a GPU for a moment, so the first one may wait in
a short queue.
