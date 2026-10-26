"""The two models behind the service, as plain Python objects — no web code (M6).

    TextGenerator   a prompt -> text, streamed piece by piece           (gpt-llama, M3/M4)
    Parser          Hinglish requests -> intent and slots, many at once  (M5's best parser)

The same objects run inside the FastAPI server (each behind a worker thread, worker.py) and inside
the Hugging Face Space (called from @spaces.GPU functions), so both serve exactly the same code.
"""
from __future__ import annotations

import codecs
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import torch

from ..bundle import load_bundle
from ..text import normalize
from ..top import Node, from_model, parse_tree, prompt_ids, scored_form, slot_items, top_intent


# ------------------------------------------------------------------ text generation
@dataclass
class GenerateParams:
    """Decoding settings (M4). The defaults are sample.py's: temperature 0.8, top-k 50."""
    max_new_tokens: int = 120
    temperature: float = 0.8           # 0 = greedy
    top_k: int | None = 50             # None or 0 = off
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int | None = None            # None = a fresh random seed (returned, so a result can be redone)


@dataclass
class Piece:
    text: str                          # the next piece of text: complete characters only


@dataclass
class Done:
    tokens: int                        # tokens generated (the final <|endoftext|> not counted)
    finish_reason: str                 # "stop": the model ended the text | "length" | "cancelled"
    seed: int
    seconds: float                     # from the start of the request to the last token
    first_token_seconds: float | None  # prompt read + first token chosen ("time to first token")


class TextGenerator:
    def __init__(self, model, tok, config: dict, device="cpu"):
        self.model, self.tok, self.config, self.device = model, tok, config, torch.device(device)

    @classmethod
    def from_bundle(cls, path: str | Path, device="cpu") -> "TextGenerator":
        model, tok, config = load_bundle(path, device)
        return cls(model, tok, config, device)

    @property
    def info(self) -> dict:
        return {"params": self.config["params"], "task": self.config["task"], "scores": self.config["scores"],
                "source": self.config["source"], "context": self.model.cfg.block_size, "device": self.device.type}

    def stream(self, prompt: str, params: GenerateParams,
               cancelled: Callable[[], bool] = lambda: False) -> Iterator[Piece | Done]:
        """Yield the continuation of `prompt` as Pieces, then one Done.

        Like sampling.stream (M4): the prompt starts after <|endoftext|> ("a document starts here"),
        generation stops at the next <|endoftext|>, and an incremental UTF-8 decoder holds back the
        bytes of a Devanagari letter split between two tokens. `cancelled` is checked after every
        token, so a client that disconnects stops the work instead of letting it run on.
        """
        t0 = time.perf_counter()
        seed = params.seed if params.seed is not None else secrets.randbelow(2**31)
        rng = torch.Generator().manual_seed(seed)      # sampling runs on the CPU (iter_tokens)
        x = torch.tensor([[self.tok.eot_id] + self.tok.encode_ordinary(prompt)], device=self.device)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n, first, reason = 0, None, "length"
        tokens = self.model.iter_tokens(
            x, params.max_new_tokens, eot_id=self.tok.eot_id, generator=rng, temperature=params.temperature,
            top_k=params.top_k or None, top_p=params.top_p, min_p=params.min_p,
            repetition_penalty=params.repetition_penalty)
        for nxt in tokens:
            if first is None:
                first = time.perf_counter() - t0
            t = int(nxt[0, 0])
            if t == self.tok.eot_id:
                reason = "stop"
                break
            n += 1
            piece = decoder.decode(self.tok.decode_bytes([t]))
            if piece:
                yield Piece(piece)
            if cancelled():
                reason = "cancelled"
                break
        tokens.close()                                 # finish the model's generator now
        tail = decoder.decode(b"", final=True)
        if tail:
            yield Piece(tail)
        yield Done(n, reason, seed, time.perf_counter() - t0, first)


# ------------------------------------------------------------------ parsing
@dataclass
class ParseResult:
    text: str                          # the request, normalised as in training
    written: str                       # what the model wrote (labels in upper case again)
    parse: str                         # the decoupled tree: what an assistant acts on, what M5 scored
    valid: bool                        # is `parse` a well-formed tree?
    intent: str | None                 # e.g. "CREATE_ALARM"
    slots: list[dict] = field(default_factory=list)    # [{"label": "DATE_TIME", "text": "kal subah 7 baje"}]
    spans: list[dict] = field(default_factory=list)    # the written words, each run with its slot (or None)
    tokens: int = 0                    # tokens the model wrote
    batch_size: int = 1                # how many requests shared this forward pass
    error: str | None = None           # why the request was not parsed (empty, too long)

    def to_dict(self) -> dict:
        return asdict(self)


def word_spans(tree: Node) -> list[dict]:
    """Every word of an annotated (full) tree with the innermost slot it sits in, neighbours with the
    same slot merged: [{"text": "kal subah 7 baje", "label": "DATE_TIME"}, {"text": "alarm laga do",
    "label": None}]. Shows *where* in the request each slot is — what a highlighter needs."""
    flat: list[tuple[str, str | None]] = []

    def walk(node: Node, slot: str | None) -> None:
        here = node.label[3:] if node.label.startswith("SL:") else slot
        for c in node.children:
            if isinstance(c, str):
                flat.append((c, here))
            else:
                walk(c, here)

    walk(tree, None)
    spans: list[dict] = []
    for word, label in flat:
        if spans and spans[-1]["label"] == label:
            spans[-1]["text"] += " " + word
        else:
            spans.append({"text": word, "label": label})
    return spans


@torch.no_grad()
def greedy_batch(model, prompts: list[list[int]], eot_id: int, pad_id: int, max_new_tokens: int,
                 device="cpu") -> list[list[int]]:
    """Greedy continuations of token prompts of *different* lengths, all in one batch: shorter
    prompts are padded on the left (models/gpt.py masks the padding), and each row stops at
    <|endoftext|> — the batch stops when every row has. Returns each row's tokens before it."""
    T = max(len(p) for p in prompts)
    x = torch.tensor([[pad_id] * (T - len(p)) + p for p in prompts], device=device)
    pad = torch.tensor([T - len(p) for p in prompts])
    n_new = max(1, min(max_new_tokens, model.cfg.block_size - T + 1))
    done = torch.zeros(len(prompts), dtype=torch.bool)
    steps = []
    for nxt in model.iter_tokens(x, n_new, temperature=0, pad=pad):
        nxt = nxt[:, 0].cpu()
        steps.append(nxt)
        done |= nxt == eot_id
        if bool(done.all()):
            break
    rows = torch.stack(steps, dim=1).tolist()
    return [row[:row.index(eot_id)] if eot_id in row else row for row in rows]


class Parser:
    """M5's semantic parser. parse() takes any number of requests and decodes them as one batch."""

    def __init__(self, model, tok, config: dict, device="cpu", max_request_tokens: int = 64,
                 max_new_tokens: int = 192):
        task = config["task"]
        if task.get("name") != "hinglish-top":
            raise ValueError(f"not a parser bundle: task {task}")
        self.model, self.tok, self.config, self.device = model, tok, config, torch.device(device)
        self.labels, self.target = task["labels"], task["target"]
        self.max_request_tokens, self.max_new_tokens = max_request_tokens, max_new_tokens
        self.pad_id = tok.special_tokens["<|pad|>"]

    @classmethod
    def from_bundle(cls, path: str | Path, device="cpu", **kwargs) -> "Parser":
        model, tok, config = load_bundle(path, device)
        return cls(model, tok, config, device, **kwargs)

    @property
    def info(self) -> dict:
        return {"params": self.config["params"], "task": self.config["task"], "scores": self.config["scores"],
                "source": self.config["source"], "max_request_tokens": self.max_request_tokens,
                "device": self.device.type}

    def prompt(self, text: str) -> list[int]:
        """Token ids the model reads for one request. Raises ValueError for an empty request or one
        that is too long (only 1 of Hinglish-TOP's 10,896 requests is): the parse must still fit in
        the context window."""
        text = normalize(text)
        if not text:
            raise ValueError("the request is empty")
        ids = prompt_ids(self.tok, text)
        if len(ids) > self.max_request_tokens:
            raise ValueError(f"the request is {len(ids)} tokens long; the limit is {self.max_request_tokens}")
        return ids

    def parse(self, texts: list[str]) -> list[ParseResult]:
        """Parse every request in one batch. A request that can't be parsed (empty, too long) gets a
        result with `error` set, so it never fails the requests it was batched with."""
        results: list[ParseResult | None] = [None] * len(texts)
        prompts, rows = [], []
        for i, t in enumerate(texts):
            try:
                prompts.append(self.prompt(t))
                rows.append(i)
            except ValueError as e:
                results[i] = ParseResult(normalize(t), "", "", False, None, error=str(e))
        if prompts:
            outs = greedy_batch(self.model, prompts, self.tok.eot_id, self.pad_id, self.max_new_tokens,
                                self.device)
            for i, out in zip(rows, outs):
                results[i] = self._result(normalize(texts[i]), out, len(prompts))
        return results

    def _result(self, text: str, out: list[int], batch_size: int) -> ParseResult:
        written = from_model(self.tok.decode(out).strip(), self.labels)
        annotated = self.target == "annotated"
        parse = scored_form(written) if annotated else written
        result = ParseResult(text, written, parse, False, None, tokens=len(out), batch_size=batch_size)
        try:
            tree = parse_tree(parse)
        except ValueError:
            return result                              # not a tree: shown as written, valid = False
        result.valid, result.intent = True, top_intent(parse)[3:]
        result.slots = [{"label": label[3:], "text": words} for label, words in slot_items(tree)]
        if annotated:
            result.spans = word_spans(parse_tree(written))
        return result
