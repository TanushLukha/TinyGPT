"""Corpus sources for TinyGPT. Every source yields cleaned documents as plain strings.

full corpus (needs internet + `pip install -e ".[data]"`):
  * Hindi Wikipedia            HF  wikimedia/wikipedia, config 20231101.hi   (CC BY-SA)
  * Hinglish translations      HF  findnitai/english-to-hinglish             (Apache-2.0)
  * Hinglish-TOP queries       GitHub google-research-datasets               (Apache-2.0)
dev corpus (small, GitHub only — for quick tests):
  * UD Hindi HDTB sentences    GitHub UniversalDependencies (CC BY-NC-SA, dev use only)
  * Hinglish-TOP (human-annotated part)
"""
from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from .text import normalize, script_shares, strip_wiki_tail

TOP_BASE = "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/main/Dataset/"
TOP_FILES = {
    "human": ["Human%20Annotated%20Data/train.tsv", "Human%20Annotated%20Data/validation.tsv",
              "Human%20Annotated%20Data/test.tsv"],
    "synthetic": ["Synthetically%20Generated%20Data/train.tsv"],
}
UD_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master/"
UD_FILES = ["hi_hdtb-ud-train.conllu", "hi_hdtb-ud-dev.conllu", "hi_hdtb-ud-test.conllu"]


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8")


def _group(sentences: Iterator[str], per_doc: int) -> Iterator[str]:
    """Drop repeated sentences, then join consecutive ones into documents of `per_doc` lines."""
    buf: list[str] = []
    seen: set[str] = set()
    for s in sentences:
        if s in seen:
            continue
        seen.add(s)
        buf.append(s)
        if len(buf) == per_doc:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


# ---------------------------------------------------------------- Hindi
def hindi_wikipedia(max_bytes: int, min_chars: int = 300, min_devanagari: float = 0.6,
                    seed: int = 0) -> Iterator[str]:
    """Stream Hindi Wikipedia articles (shuffled), cleaned, until max_bytes of text."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    total = 0
    for row in ds:
        text = normalize(strip_wiki_tail(row["text"]))
        if len(text) < min_chars or script_shares(text)[0] < min_devanagari:
            continue
        total += len(text.encode("utf-8"))
        yield text
        if total >= max_bytes:
            return


def ud_hindi(per_doc: int = 8) -> Iterator[str]:
    """Dev corpus: the ~16.6k news sentences of the UD Hindi treebank."""
    def sentences():
        for name in UD_FILES:
            for line in _fetch(UD_BASE + name).splitlines():
                if line.startswith("# text = "):
                    s = normalize(line[len("# text = "):])
                    if s:
                        yield s
    yield from _group(sentences(), per_doc)


# ---------------------------------------------------------------- Hinglish
def _hinglish_ok(s: str, min_latin: float = 0.8) -> bool:
    return len(s) >= 10 and script_shares(s)[1] >= min_latin


def hinglish_top(parts: tuple[str, ...] = ("human", "synthetic"), per_doc: int = 16) -> Iterator[str]:
    """Code-switched assistant queries ("kal ka weather kaisa hoga") from Hinglish-TOP."""
    def sentences():
        for part in parts:
            for name in TOP_FILES[part]:
                reader = csv.DictReader(io.StringIO(_fetch(TOP_BASE + name)), delimiter="\t",
                                        quoting=csv.QUOTE_NONE)
                for row in reader:
                    s = normalize(row.get("cs_query") or "")
                    if _hinglish_ok(s):
                        yield s
    yield from _group(sentences(), per_doc)


def _pick_hinglish(example: dict) -> str | None:
    """findnitai/english-to-hinglish stores pairs; find the Hinglish side robustly."""
    if isinstance(example.get("translation"), dict):
        t = example["translation"]
        for key in ("hi_ng", "hinglish", "hi_en", "cs"):
            if key in t:
                return t[key]
    for key in ("hi_ng", "hinglish", "Hinglish", "target", "output"):
        if isinstance(example.get(key), str):
            return example[key]
    return None


def findnitai_hinglish(per_doc: int = 16) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset("findnitai/english-to-hinglish", split="train")

    def sentences():
        for ex in ds:
            s = _pick_hinglish(ex)
            if s is None:
                raise KeyError(f"no Hinglish field found in columns {list(ex)}; update _pick_hinglish")
            s = normalize(s)
            if _hinglish_ok(s):
                yield s
    yield from _group(sentences(), per_doc)


# ---------------------------------------------------------------- reading back
HINDI_SOURCES = {"wiki_hi", "ud_hindi"}
HINGLISH_SOURCES = {"findnitai", "hinglish_top"}


def read_jsonl(path: str | Path, sources: set[str] | None = None) -> Iterator[tuple[str, str]]:
    """Yield (text, source) from a data/clean/*.jsonl file, optionally for some sources only."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if sources is None or rec["src"] in sources:
                yield rec["text"], rec["src"]
