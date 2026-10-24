import random
from collections import Counter

import pytest
import regex

from tinygpt.bpe import BPETokenizer, GPT4_PATTERN, INDIC_PATTERN, merge_pair

HINDI = "भारत एक विशाल देश है। हिन्दी भारत की राजभाषा है और करोड़ों लोग इसे बोलते हैं।"
HINGLISH = "kal mai office jaunga, phir hum movie dekhenge. Weekend pe plan karte hai!"


def naive_train(texts, vocab_size, pattern="indic", min_frequency=2):
    """Slow reference trainer: recount every pair after every merge."""
    rx = regex.compile({"indic": INDIC_PATTERN, "gpt4": GPT4_PATTERN}[pattern])
    counts = Counter()
    for t in texts:
        counts.update(rx.findall(t))
    words = [(list(w.encode("utf-8")), f) for w, f in counts.items()]
    merges = []
    for k in range(vocab_size - 256):
        stats = Counter()
        for w, f in words:
            for p in zip(w, w[1:]):
                stats[p] += f
        if not stats:
            break
        best = min(stats, key=lambda p: (-stats[p], p))     # most frequent, then smallest pair
        if stats[best] < min_frequency:
            break
        merges.append(best)
        words = [(merge_pair(w, best, 256 + k), f) for w, f in words]
    return merges


def test_merge_pair():
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge_pair([5, 5, 5], (5, 5), 7) == [7, 5]              # left to right, no overlap
    assert merge_pair([], (1, 2), 9) == []


def test_wikipedia_example():
    # https://en.wikipedia.org/wiki/Byte_pair_encoding : aaabdaaabac -> aa, ab, aaab
    tok = BPETokenizer.train(["aaabdaaabac"], vocab_size=256 + 3, special_tokens=[])
    assert [tok.token_str(256 + r) for r in range(3)] == ["aa", "ab", "aaab"]
    assert tok.encode("aaabdaaabac") == [258, ord("d"), 258, ord("a"), ord("c")]


@pytest.mark.parametrize("seed", range(25))
def test_fast_trainer_matches_naive(seed):
    rnd = random.Random(seed)
    alphabet = "abcक्षिमाें "
    texts = ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 60))) for _ in range(12)]
    vocab = 256 + rnd.randint(1, 40)
    for pattern in ("indic", "gpt4"):
        fast = BPETokenizer.train(texts, vocab, pattern=pattern, special_tokens=[], min_frequency=2)
        assert fast.merges == naive_train(texts, vocab, pattern=pattern, min_frequency=2)


def test_roundtrip_unicode():
    rnd = random.Random(0)
    tok = BPETokenizer.train([HINDI * 3, HINGLISH * 3], vocab_size=400)
    pool = list("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह ािीुूेैोौंँः़्") + \
        list("abcxyzABC0123456789 .,!?-'\"\n\t") + ["।", "\u200d", "\u200c", "😀", "€", "ß", "中"]
    for _ in range(300):
        s = "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s
    for s in (HINDI, HINGLISH, "", " ", "\n\n", "  leading and trailing  "):
        assert tok.decode(tok.encode(s)) == s


def test_training_compresses():
    tok = BPETokenizer.train([HINDI * 20, HINGLISH * 20], vocab_size=600)
    for s in (HINDI, HINGLISH):
        assert len(tok.encode(s)) < len(s.encode("utf-8")) / 2


def test_special_tokens():
    tok = BPETokenizer.train([HINDI, HINGLISH], vocab_size=300)
    eot = tok.eot_id
    assert eot == tok.vocab_size - len(tok.special_tokens)       # specials come last
    text = f"नमस्ते<|endoftext|>hello"
    ids = tok.encode(text)
    assert ids.count(eot) == 1 and tok.decode(ids) == text
    plain = tok.encode(text, allowed_special="none")             # treated as ordinary text
    assert eot not in plain and tok.decode(plain) == text
    # special strings never take part in merges during training
    tok2 = BPETokenizer.train(["<|endoftext|>" * 50], vocab_size=300)
    assert all(b"endoftext" not in tok2.vocab[i] for i in range(256, 256 + len(tok2.merges)))


def test_save_load(tmp_path):
    tok = BPETokenizer.train([HINDI * 5, HINGLISH * 5], vocab_size=420)
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    for s in (HINDI, HINGLISH, "mixed हिन्दी text 123<|endoftext|>"):
        assert tok.encode(s) == tok2.encode(s)
    assert tok2.vocab_size == tok.vocab_size and tok2.pattern_name == "indic"
    tok.save_vocab(tmp_path / "vocab.txt")
    assert (tmp_path / "vocab.txt").read_text().count("\n") == tok.vocab_size


def test_indic_pattern_keeps_hindi_words_whole():
    indic, gpt4 = regex.compile(INDIC_PATTERN), regex.compile(GPT4_PATTERN)
    assert indic.findall("हिन्दी भाषा") == ["हिन्दी", " भाषा"]
    assert len(gpt4.findall("हिन्दी")) > 1                       # split at vowel signs
    zwj_word = "मल्\u200dलों"                                      # zero-width joiner inside a word
    assert indic.findall(zwj_word) == [zwj_word]
    assert indic.findall("Hello, दुनिया!") == ["Hello", ",", " दुनिया", "!"]
    assert indic.findall("सन् 1947 में।") == ["सन्", " ", "194", "7", " में", "।"]


def test_unknown_id_raises():
    tok = BPETokenizer.train([HINDI], vocab_size=300)
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 5])
