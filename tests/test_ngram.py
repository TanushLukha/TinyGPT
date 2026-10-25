from collections import Counter

import numpy as np
import pytest

from tinygpt.models.ngram import NGramLM, Smoothing, ngram_codes

V, EOT = 7, 6


def corpus(n=400, seed=0):
    return np.random.default_rng(seed).integers(0, V, n).astype(np.uint16)


def test_codes():
    t = np.array([1, 2, 3, 4])
    assert ngram_codes(t, 1, 10).tolist() == [1, 2, 3, 4]
    assert ngram_codes(t, 2, 10).tolist() == [12, 23, 34]
    assert ngram_codes(t, 3, 10).tolist() == [123, 234]
    with pytest.raises(ValueError):
        ngram_codes(t, 5, 8192)                           # 8192^5 does not fit in int64


def test_counts_match_counter():
    t = corpus()
    lm = NGramLM(3, V, EOT).fit(t)
    for n in (1, 2, 3):
        want = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
        table = lm.tables[n - 1]
        got = {}
        for code, c in zip(table.codes, table.counts):
            gram = []
            for _ in range(n):
                gram.append(int(code % V))
                code //= V
            got[tuple(reversed(gram))] = int(c)
        assert got == {tuple(int(x) for x in k): v for k, v in want.items()}


SMOOTHINGS = [Smoothing(1, "addk", k=0.5), Smoothing(2, "addk", k=0.1), Smoothing(1, "jm", lambdas=(0.9,)),
              Smoothing(2, "jm", lambdas=(0.9, 0.6)), Smoothing(3, "jm", lambdas=(0.9, 0.6, 0.4))]


@pytest.mark.parametrize("sm", SMOOTHINGS, ids=lambda s: f"{s.kind}{s.order}")
def test_distributions_sum_to_one(sm):
    lm = NGramLM(3, V, EOT).fit(corpus())
    rng = np.random.default_rng(1)
    for _ in range(20):
        hist = rng.integers(0, V, rng.integers(0, 4)).tolist()
        p = lm.next_dist(hist, sm)
        assert p.min() > 0 and p.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("sm", SMOOTHINGS, ids=lambda s: f"{s.kind}{s.order}")
def test_vectorised_scores_match_full_distribution(sm):
    """probs(stats(...)) used for evaluation == next_dist(...)[target] used for sampling."""
    lm = NGramLM(3, V, EOT).fit(corpus())
    text = corpus(60, seed=2)
    for start, end in [(0, 60), (25, 60), (1, 10)]:
        p = lm.probs(lm.stats(text, start, end), sm)
        want = [lm.next_dist(text[:i], sm)[text[i]] for i in range(start + 1, end)]
        np.testing.assert_allclose(p, want)


def test_add_k_bigram_by_hand():
    t = np.array([0, 1, 0, 1, 0, 2])        # bigrams: 01 10 01 10 02 -> c(0,1)=2, c(0,2)=1, c(0)=3
    lm = NGramLM(2, 3, 2).fit(t)
    p = lm.next_dist([0], Smoothing(2, "addk", k=1.0))
    np.testing.assert_allclose(p, [(0 + 1) / 6, (2 + 1) / 6, (1 + 1) / 6])


def test_unseen_context_falls_back():
    t = np.array([0, 1, 0, 1, 0, 1])
    lm = NGramLM(2, 3, 2).fit(t)
    jm = Smoothing(2, "jm", lambdas=(0.5, 0.8))
    np.testing.assert_allclose(lm.next_dist([2], jm), lm.next_dist([], Smoothing(1, "jm", lambdas=(0.5,))))


def test_generate():
    lm = NGramLM(2, V, EOT).fit(corpus())
    out = lm.generate([1, 2], 30, Smoothing(2, "addk", k=0.1), np.random.default_rng(0))
    assert out[:2] == [1, 2] and len(out) <= 32
    assert EOT not in out[2:-1]                            # stops at the first <|endoftext|>
