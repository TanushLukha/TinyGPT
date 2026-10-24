"""Text clean-up shared by the data pipeline and the tokenizer scripts."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import regex

# Zero-width non-joiner / joiner change how Devanagari conjuncts are drawn: keep them.
KEEP_FORMAT = {"\u200c", "\u200d"}
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# letters AND combining marks: a Hindi vowel sign (ि, ा, ्) is a mark, not a letter
_LETTER = regex.compile(r"[\p{L}\p{M}]")
_DEVANAGARI = regex.compile(r"[[\p{L}\p{M}]&&\p{Script=Devanagari}]", regex.V1)
_LATIN = regex.compile(r"[[\p{L}\p{M}]&&\p{Script=Latin}]", regex.V1)


def normalize(text: str) -> str:
    """NFC-normalise and tidy whitespace.

    NFC matters for Hindi: the same letter can be typed as one code point or as base +
    combining mark (e.g. nukta forms). Without normalising, the tokenizer would learn
    two different spellings of the same word.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # drop control / invisible format characters (BOM, soft hyphen, zero-width space ...)
    text = "".join(ch for ch in text
                   if ch == "\n" or ch in KEEP_FORMAT or not unicodedata.category(ch).startswith("C"))
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def script_shares(text: str) -> tuple[float, float]:
    """(Devanagari share, Latin share) among the letters of `text`."""
    letters = len(_LETTER.findall(text))
    if letters == 0:
        return 0.0, 0.0
    return len(_DEVANAGARI.findall(text)) / letters, len(_LATIN.findall(text)) / letters


# Section headings after which Wikipedia articles are only lists of links / references.
WIKI_TAIL_HEADINGS = {
    "सन्दर्भ", "संदर्भ", "सन्दर्भ सूची", "संदर्भ सूची", "बाहरी कड़ियाँ", "बाहरी कड़ियां",
    "इन्हें भी देखें", "इन्हे भी देखें", "टीका टिप्पणी", "टिप्पणियाँ", "स्रोत", "ग्रन्थसूची",
    "References", "External links", "See also", "Notes",
}


def strip_wiki_tail(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() in WIKI_TAIL_HEADINGS:
            return "\n".join(lines[:i])
    return text


def is_val(text: str, val_permille: int) -> bool:
    """Deterministic train/val split: the same document always lands in the same split."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return h % 1000 < val_permille
