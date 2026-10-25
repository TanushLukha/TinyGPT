"""Turning the model's next-token scores (logits) into text.

Decoding strategies, applied in this order to the logits of the next position:
  repetition penalty  push down tokens that already appeared (CTRL, 2019)
  temperature         divide logits by T: < 1 sharper (safer), > 1 flatter (more varied); 0 = greedy
  top-k               keep the k most likely tokens
  top-p (nucleus)     keep the smallest set of tokens whose probabilities add up to p (2019)
  min-p               keep tokens at least min_p times as likely as the best one (2024)
then sample from what is left.
"""
from __future__ import annotations

import codecs
from typing import Iterator

import torch


def filter_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, *,
                  top_p: float | None = None, min_p: float | None = None, repetition_penalty: float = 1.0,
                  history: torch.Tensor | None = None) -> torch.Tensor:
    """Every strategy except the final draw: logits (B, V) -> new logits, -inf for removed tokens.

    `temperature` must be > 0; `history` (B, T) holds the tokens so far (for the repetition penalty).
    """
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and history is not None and history.numel():
        seen = logits.gather(1, history)
        logits.scatter_(1, history, torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty))
    logits = logits / temperature
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        probs, order = torch.softmax(logits, dim=-1).sort(dim=-1, descending=True)
        drop = probs.cumsum(dim=-1) - probs > top_p        # everything after the set reaching top_p
        logits = logits.masked_fill(drop.scatter(1, order, drop), float("-inf"))
    if min_p is not None and min_p > 0:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(dim=-1, keepdim=True).values, float("-inf"))
    return logits


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                generator: torch.Generator | None = None, *, top_p: float | None = None,
                min_p: float | None = None, repetition_penalty: float = 1.0,
                history: torch.Tensor | None = None) -> torch.Tensor:
    """logits (B, V) for the next position -> (B, 1) chosen token ids. `history` (B, T): tokens so far."""
    if temperature == 0:                                   # greedy: the filters can't change the argmax
        return filter_logits(logits, repetition_penalty=repetition_penalty,
                             history=history).argmax(dim=-1, keepdim=True)
    logits = filter_logits(logits, temperature, top_k, top_p=top_p, min_p=min_p,
                           repetition_penalty=repetition_penalty, history=history)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator)


def complete(model, tok, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
             top_k: int | None = 50, device="cpu", generator: torch.Generator | None = None,
             **sampling) -> str:
    """Prompt -> prompt + continuation, as text.

    The prompt is preceded by <|endoftext|>: in training, that token always comes right before a
    new document, so it tells the model "a document starts here". Generation stops at the next
    <|endoftext|> (end of the document) or after max_new_tokens.
    """
    return prompt + "".join(stream(model, tok, prompt, max_new_tokens, temperature=temperature, top_k=top_k,
                                   device=device, generator=generator, **sampling))


def stream(model, tok, prompt: str, max_new_tokens: int = 100, device="cpu",
           generator: torch.Generator | None = None, **sampling) -> Iterator[str]:
    """Yield the continuation of `prompt` piece by piece, as it is generated.

    A token is a sequence of *bytes*, and it can end in the middle of a character: every Devanagari
    letter is 3 bytes in UTF-8, and a token may hold only the first 2. Decoding each token on its own
    would print a "�". An incremental decoder keeps incomplete bytes until the character is whole.
    """
    ids = [tok.eot_id] + tok.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for nxt in model.iter_tokens(x, max_new_tokens, eot_id=tok.eot_id, generator=generator, **sampling):
        t = int(nxt[0, 0])
        if t == tok.eot_id:
            break
        piece = decoder.decode(tok.decode_bytes([t]))
        if piece:
            yield piece
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
