"""Hinglish-TOP: semantic parsing of code-switched (Hindi-English) assistant requests — M5's task.

    query:  do ghante ke liye alarm set kardo
    parse:  [IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] ]

The human-annotated part of Hinglish-TOP (Agarwal et al., 2022; Apache-2.0): 2,993 training, 1,390
validation and 6,513 test requests in 8 domains (alarm, event, messaging, music, navigation,
reminder, timer, weather). A parse is a tree: an intent (IN:) holds words and slots (SL:), a slot
holds words and, sometimes, a nested intent. We predict the *decoupled* tree (Aghajanyan et al.,
2020), which keeps only the words inside slots — the information an assistant acts on:

    full:       [IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] alarm set kardo ]
    decoupled:  [IN:CREATE_ALARM [SL:DATE_TIME do ghante ke liye ] ]

As a language-modelling problem: "<|endoftext|>{query} => {parse}<|endoftext|>", with the loss only
on the parse (everything after "=>"). The model may write either form (`target`): "decoupled", or
"annotated" — the full tree, i.e. the request itself with brackets inserted, which turns finding the
slot words into copying the request in order. Either way it is scored on the decoupled tree.
"""
from __future__ import annotations

import csv
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import TOP_BASE
from .text import normalize

FILES = {"train": "train", "val": "validation", "test": "test"}
SEP = " =>"                     # between the request and its parse


@dataclass(frozen=True)
class Example:
    query: str
    parse: str                  # decoupled: what is scored
    domain: str
    full: str = ""              # the full tree: the request with brackets


# ------------------------------------------------------------------ data
def download(data_dir: str | Path = "data/top") -> Path:
    """Fetch the three human-annotated splits once (≈3 MB) into data_dir/{train,val,test}.tsv."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, name in FILES.items():
        path = out / f"{split}.tsv"
        if not path.exists():
            url = TOP_BASE + f"Human%20Annotated%20Data/{name}.tsv"
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
    return out


def load(split: str, data_dir: str | Path = "data/top") -> list[Example]:
    path = download(data_dir) / f"{split}.tsv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return [Example(normalize(r["cs_query"]), decoupled(normalize(r["cs_parse"])), r["domain"],
                    to_string(parse_tree(normalize(r["cs_parse"])))) for r in rows]


def match_key(text: str) -> str:
    """Loose form for finding the same request again: case, punctuation and spacing ignored."""
    return " ".join(re.sub(r"[^\w\s]", " ", normalize(text).lower()).split())


# ------------------------------------------------------------------ trees
@dataclass
class Node:
    label: str                  # "IN:CREATE_ALARM" or "SL:DATE_TIME"
    children: list              # words (str) and nodes, in order


def parse_tree(s: str) -> Node:
    """'[IN:X word [SL:Y word ] ]' -> Node. Raises ValueError for anything that is not a valid tree:
    unbalanced brackets, words outside the root, a root that is not an intent, an intent directly
    inside an intent or a slot directly inside a slot."""
    stack: list[Node] = []
    root = None
    for tok in s.split():
        if tok.startswith("[") and len(tok) > 1:
            node = Node(tok[1:], [])
            kind = node.label[:3]
            if kind not in ("IN:", "SL:") or len(node.label) == 3:
                raise ValueError(f"bad label {tok!r}")
            if stack:
                if stack[-1].label[:3] == kind:
                    raise ValueError(f"{node.label} directly inside {stack[-1].label}")
                stack[-1].children.append(node)
            elif root is None:
                if kind != "IN:":
                    raise ValueError("the root must be an intent")
                root = node
            else:
                raise ValueError("more than one root")
            stack.append(node)
        elif tok == "]":
            if not stack:
                raise ValueError("unbalanced ']'")
            stack.pop()
        else:
            if not stack:
                raise ValueError(f"word {tok!r} outside the tree")
            stack[-1].children.append(tok)
    if root is None or stack:
        raise ValueError("empty or unclosed tree")
    return root


def to_string(node: Node) -> str:
    return " ".join(["[" + node.label, *(c if isinstance(c, str) else to_string(c) for c in node.children), "]"])


def decouple(node: Node) -> Node:
    """Drop the words that sit directly in an intent; keep slots and everything inside them."""
    if node.label.startswith("IN:"):
        return Node(node.label, [decouple(c) for c in node.children if isinstance(c, Node)])
    return Node(node.label, [c if isinstance(c, str) else decouple(c) for c in node.children])


def decoupled(parse: str) -> str:
    return to_string(decouple(parse_tree(parse)))


def scored_form(pred: str) -> str:
    """A predicted full tree, decoupled for scoring; a prediction that is not a tree stays as it is
    (and is scored as invalid)."""
    try:
        return decoupled(pred)
    except ValueError:
        return pred


def words(node: Node) -> list[str]:
    return [w for c in node.children for w in ([c] if isinstance(c, str) else words(c))]


def slot_items(node: Node) -> list[tuple[str, str]]:
    """(slot label, slot text) for every slot in the tree, nested ones included."""
    out = []
    for c in node.children:
        if isinstance(c, Node):
            if c.label.startswith("SL:"):
                out.append((c.label, " ".join(words(c))))
            out.extend(slot_items(c))
    return out


def top_intent(parse: str) -> str | None:
    first = parse.split()[:1]
    return first[0][1:] if first and first[0].startswith("[IN:") else None


# ------------------------------------------------------------------ metrics
def score(preds: list[str], golds: list[str]) -> dict:
    """exact_match: the whole tree is right; intent_acc: the top intent is right; slot_p/r/f1: over
    (slot label, slot text) pairs, micro-averaged; valid: share of predictions that are trees."""
    n = len(golds)
    em = sum(" ".join(p.split()) == " ".join(g.split()) for p, g in zip(preds, golds))
    intent = sum(top_intent(p) is not None and top_intent(p) == top_intent(g) for p, g in zip(preds, golds))
    tp = n_pred = n_gold = valid = 0
    for p, g in zip(preds, golds):
        gold = Counter(slot_items(parse_tree(g)))
        try:
            pred = Counter(slot_items(parse_tree(p)))
            valid += 1
        except ValueError:
            pred = Counter()
        tp += sum((gold & pred).values())
        n_pred += sum(pred.values())
        n_gold += sum(gold.values())
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {"n": n, "exact_match": em / n, "intent_acc": intent / n, "slot_p": prec, "slot_r": rec,
            "slot_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "valid": valid / n}


# ------------------------------------------------------------------ as a language-modelling task
def to_model(parse: str, labels: str = "lower") -> str:
    """The parse as the model writes it. Our tokenizer spells upper-case labels letter by letter
    ("[IN:CREATE_ALARM" = 14 tokens); in lower case they are shorter and share pieces with words the
    model knows ("[in:create_alarm" = 8). The mapping is exact both ways: every label is upper-case."""
    if labels == "upper":
        return parse
    return " ".join(t.lower() if t.startswith("[") else t for t in parse.split())


def from_model(text: str, labels: str = "lower") -> str:
    if labels == "upper":
        return text
    return " ".join(t.upper() if t.startswith("[") else t for t in text.split())


def prompt_ids(tok, query: str) -> list[int]:
    return [tok.eot_id] + tok.encode_ordinary(query) + tok.encode_ordinary(SEP)


def target_text(ex: Example, target: str = "decoupled") -> str:
    return ex.full if target == "annotated" else ex.parse


def example_ids(tok, ex: Example, labels: str = "lower", target: str = "decoupled") -> tuple[list[int], int]:
    """(token ids of the whole example, number of prompt tokens). The parse starts with a space,
    as words do after "=>"; the example ends with <|endoftext|> so the model learns to stop."""
    prompt = prompt_ids(tok, ex.query)
    out = tok.encode_ordinary(" " + to_model(target_text(ex, target), labels))
    return prompt + out + [tok.eot_id], len(prompt)


def fewshot_prompt_ids(tok, demos: list[Example], query: str, labels: str = "lower",
                       target: str = "decoupled") -> list[int]:
    """No training: solved examples, one per line, then the new request (in-context learning)."""
    ids = [tok.eot_id]
    for d in demos:
        ids += tok.encode_ordinary(d.query + SEP + " " + to_model(target_text(d, target), labels) + "\n")
    return ids + tok.encode_ordinary(query) + tok.encode_ordinary(SEP)


def make_batch(items: list[tuple[list[int], int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded inputs x and targets y (= x shifted by one). Targets inside the prompt and in the
    padding are -100, which cross-entropy ignores: the loss is only on the parse and the final
    <|endoftext|>. Causal attention never looks right, so padding cannot change any real position."""
    T = max(len(ids) for ids, _ in items) - 1
    x = torch.full((len(items), T), pad_id, dtype=torch.long)
    y = torch.full((len(items), T), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(items):
        t = torch.tensor(ids)
        x[i, :len(ids) - 1] = t[:-1]
        y[i, :len(ids) - 1] = t[1:]
        y[i, :n_prompt - 1] = -100
    return x, y


@torch.no_grad()
def predict(model, tok, prompts: list[list[int]], device, batch_size: int = 64, max_new_tokens: int = 256,
            stop: str | None = None) -> list[str]:
    """Greedy continuation of each prompt (token ids), up to <|endoftext|> or a full context window.
    With `stop` (few-shot: a newline), the answer is the first non-empty line of the continuation.

    Prompts of the same length are batched together, so no padding is needed and every prompt is
    read exactly as in training; the KV-cache (M4) makes each new token cost one position.
    """
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    window = getattr(getattr(model, "cfg", None), "block_size", None)
    out: list[str] = [""] * len(prompts)
    for length, idxs in by_len.items():
        n_new = max_new_tokens if window is None else max(1, min(max_new_tokens, window - length))
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            x = torch.tensor([prompts[i] for i in chunk], device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool)
            steps = []
            for nxt in model.iter_tokens(x, n_new, temperature=0):
                nxt = nxt[:, 0].cpu()
                steps.append(nxt)
                done |= nxt == tok.eot_id             # every sequence has finished: stop early
                if bool(done.all()):
                    break
            gen = torch.stack(steps, dim=1).tolist()
            for i, row in zip(chunk, gen):
                text = tok.decode(row[:row.index(tok.eot_id)] if tok.eot_id in row else row)
                if stop:
                    text = next((line for line in text.split(stop) if line.strip()), "")
                out[i] = text.strip()
    return out
