"""Byte-level BPE tokenizer, written from scratch for Hindi + Hinglish text.

How it works, in four steps:
  1. Pre-tokenise: a regex cuts text into chunks (words, numbers, punctuation, spaces).
     Merges never cross a chunk boundary, so "a dog" can never become one token.
  2. Bytes: each chunk becomes its UTF-8 bytes, so the base vocabulary is just the
     256 byte values — any string can be encoded, nothing is ever "unknown".
  3. Train: repeatedly find the most frequent adjacent pair of tokens and merge it
     into a new token. Merge number r gets id 256 + r.
  4. Encode: apply the learned merges to new text, lowest rank (= learned earliest) first.

The Devanagari detail: GPT-2/GPT-4 regexes match words with \\p{L}+ (letters only).
Hindi vowel signs (ि ी ु े ो ं ्) are combining MARKS (\\p{M}), not letters, so those
regexes chop "हिन्दी" into pieces at every vowel sign and no token can ever span a
whole syllable. The "indic" pattern below matches [\\p{L}\\p{M}]+ instead.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import regex

# GPT-4 (cl100k_base) pre-tokeniser, for comparison. Letters only: \p{L}.
GPT4_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Same structure, but a "word" is letters AND combining marks (Devanagari vowel signs,
# virama, nukta, anusvara) plus the zero-width (non-)joiners used inside Hindi words.
_W = r"\p{L}\p{M}\u200c\u200d"
INDIC_PATTERN = (
    rf"""'(?i:[sdmt]|ll|ve|re)|[^\r\n{_W}\p{{N}}]?+[{_W}]+|\p{{N}}{{1,3}}| ?[^\s{_W}\p{{N}}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

PATTERNS = {"gpt4": GPT4_PATTERN, "indic": INDIC_PATTERN}

# Special tokens get the ids right after the merges. Reserving a few now means the
# vocabulary doesn't change when fine-tuning needs chat or task markers later (M5).
DEFAULT_SPECIALS = ["<|endoftext|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(6)]


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with `new_id`."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], pattern: str = "indic",
                 special_tokens: list[str] | None = None, meta: dict | None = None):
        self.pattern_name = pattern if pattern in PATTERNS else "custom"
        self.pattern = PATTERNS.get(pattern, pattern)
        self._rx = regex.compile(self.pattern)
        self.merges = [tuple(p) for p in merges]
        self.ranks = {p: r for r, p in enumerate(self.merges)}       # pair -> rank
        self.vocab = {i: bytes([i]) for i in range(256)}              # id -> bytes
        for r, (a, b) in enumerate(self.merges):
            self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
        specials = DEFAULT_SPECIALS if special_tokens is None else special_tokens
        base = 256 + len(self.merges)
        self.special_tokens = {s: base + i for i, s in enumerate(specials)}
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        self._special_rx = (regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
                            if specials else None)
        self.meta = meta or {}
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ properties
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, pattern: str = "indic",
              special_tokens: list[str] | None = None, min_frequency: int = 2,
              verbose: bool = False) -> "BPETokenizer":
        """Learn merges from `texts`. vocab_size counts bytes + merges + special tokens.

        Speed trick: pre-tokenised chunks repeat a lot, so we train on the multiset
        of DISTINCT chunks with their counts. A pair count is the sum of the counts
        of the chunks containing it. After each merge we only update the chunks that
        contained the merged pair (found via an index pair -> chunk ids), and a heap
        gives the most frequent pair (stale heap entries are skipped when popped).
        """
        specials = DEFAULT_SPECIALS if special_tokens is None else special_tokens
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(specials)}")
        rx = regex.compile(PATTERNS.get(pattern, pattern))
        split_rx = (regex.compile("|".join(regex.escape(s) for s in specials)) if specials else None)

        chunk_counts: Counter[str] = Counter()
        n_bytes = 0
        for text in texts:
            n_bytes += len(text.encode("utf-8"))
            parts = split_rx.split(text) if split_rx else [text]   # specials never merge
            for part in parts:
                chunk_counts.update(rx.findall(part))

        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())
        stats: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                stats[p] += f
                where[p].add(i)
        heap = [(-c, p) for p, c in stats.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        progress = None
        if verbose:
            from tqdm import tqdm
            progress = tqdm(total=num_merges, desc="BPE merges", unit="merge")
        while len(merges) < num_merges and heap:
            neg, pair = heapq.heappop(heap)
            count = stats.get(pair, 0)
            if -neg != count:                    # stale entry: count changed since push
                if count > 0:
                    heapq.heappush(heap, (-count, pair))
                continue
            if count < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            touched: set[tuple[int, int]] = set()
            for i in where.pop(pair, ()):
                w = words[i]
                if len(w) < 2:
                    continue
                nw = merge_pair(w, pair, new_id)
                if len(nw) == len(w):            # pair no longer in this chunk
                    continue
                f = freqs[i]
                for p in zip(w, w[1:]):
                    stats[p] -= f
                    touched.add(p)
                for p in zip(nw, nw[1:]):
                    stats[p] += f
                    where[p].add(i)
                    touched.add(p)
                words[i] = nw
            stats.pop(pair, None)
            for p in touched:
                c = stats.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                elif p in stats:
                    del stats[p]
            if progress:
                progress.update(1)
        if progress:
            progress.close()
        meta = {"trained_bytes": n_bytes, "distinct_chunks": len(words),
                "requested_vocab_size": vocab_size, "min_frequency": min_frequency}
        return cls(merges, pattern=pattern, special_tokens=specials, meta=meta)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # the adjacent pair that was learned earliest (lowest rank) is merged next
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            ids = merge_pair(ids, best_pair, 256 + best_rank)
        if len(self._cache) > 500_000:          # keep memory bounded on huge corpora
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._rx.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Encode text. allowed_special: "all", "none", or a set of special strings."""
        if allowed_special == "none" or not self._special_rx:
            return self.encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        out: list[int] = []
        for part in self._special_rx.split(text):
            if part in allowed:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        # one token can end in the middle of a multi-byte character; "replace" keeps
        # decoding of arbitrary (e.g. sampled) id sequences from crashing
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Readable form of one token, for inspecting the vocabulary."""
        if i in self.inverse_special:
            return self.inverse_special[i]
        return self.vocab[i].decode("utf-8", errors="backslashreplace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "tinygpt-bpe", "version": 1,
            "pattern_name": self.pattern_name, "pattern": self.pattern,
            "special_tokens": list(self.special_tokens),
            "merges": [list(p) for p in self.merges],
            "meta": self.meta,
        }
        path.write_text(json.dumps(data, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        if data.get("type") != "tinygpt-bpe":
            raise ValueError(f"{path} is not a tinygpt-bpe tokenizer file")
        pattern = data["pattern_name"] if data["pattern_name"] in PATTERNS else data["pattern"]
        return cls([tuple(p) for p in data["merges"]], pattern=pattern,
                   special_tokens=data["special_tokens"], meta=data.get("meta"))

    def save_vocab(self, path: str | Path) -> None:
        """Human-readable vocabulary: id, token, and the two tokens it was merged from."""
        lines = [f"{i}\t{self.token_str(i)!r}" for i in range(256)]
        for r, (a, b) in enumerate(self.merges):
            i = 256 + r
            lines.append(f"{i}\t{self.token_str(i)!r}\t= {self.token_str(a)!r} + {self.token_str(b)!r}")
        for s, i in self.special_tokens.items():
            lines.append(f"{i}\t{s}\t(special)")
        Path(path).write_text("\n".join(lines) + "\n")
