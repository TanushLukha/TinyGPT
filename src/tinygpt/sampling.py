"""Turning the model's next-token scores (logits) into text."""
from __future__ import annotations

import torch


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) for the next position -> (B, 1) sampled token ids.

    temperature < 1 sharpens the distribution (safer, more repetitive), > 1 flattens it (more
    varied, more mistakes), 0 means always take the most likely token (greedy). top_k keeps only
    the k most likely tokens, cutting off the long tail of unlikely ones that produce garbage.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def complete(model, tok, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8,
             top_k: int | None = 50, device="cpu", generator: torch.Generator | None = None) -> str:
    """Prompt -> prompt + continuation, as text.

    The prompt is preceded by <|endoftext|>: in training, that token always comes right before a
    new document, so it tells the model "a document starts here". Generation stops at the next
    <|endoftext|> (end of the document) or after max_new_tokens.
    """
    ids = [tok.eot_id] + tok.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, eot_id=tok.eot_id,
                         generator=generator)[0].tolist()[1:]
    if out and out[-1] == tok.eot_id:
        out = out[:-1]
    return tok.decode(out)
