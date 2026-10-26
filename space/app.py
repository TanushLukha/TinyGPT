"""TinyGPT's Hugging Face Space: the Gradio demo on ZeroGPU hardware (M6).

scripts/deploy_hf.py uploads this file with a copy of the tinygpt package, requirements.txt,
README.md (the Space's settings) and model.json (which model revision to run). The page is the
same as `python -m tinygpt.serve`'s; only the two model calls differ: here each runs inside
@spaces.GPU, which borrows a GPU for the length of the call.
"""
import spaces  # noqa: I001 — first: ZeroGPU must patch CUDA before torch initialises it

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from tinygpt.serve.engine import GenerateParams, Parser, TextGenerator
from tinygpt.serve.ui import build_ui, theme

MODEL = json.loads(Path(__file__).with_name("model.json").read_text())       # {"repo": ..., "revision": ...}
path = Path(snapshot_download(MODEL["repo"], revision=MODEL["revision"], allow_patterns=["lm/*", "parser/*"]))

# ZeroGPU: models go to "cuda" when the app starts (a stand-in until a GPU is attached), and every
# function that uses them is decorated. Outside a ZeroGPU Space @spaces.GPU does nothing, and the
# models stay on the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
lm = TextGenerator.from_bundle(path / "lm", device)
parser = Parser.from_bundle(path / "parser", device)


@spaces.GPU(duration=20)            # at most 256 tokens: a few seconds; a short limit queues sooner
def generate(prompt: str, params: GenerateParams):
    yield from lm.stream(prompt, params)


@spaces.GPU(duration=10)
def parse(texts: list[str]):
    return parser.parse(texts)


demo = build_ui(generate, parse, lm.config, parser.config)
demo.launch(theme=theme())
