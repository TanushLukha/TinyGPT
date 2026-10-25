import torch

from tinygpt.sampling import sample_next

LOGITS = torch.log(torch.tensor([[0.50, 0.25, 0.15, 0.06, 0.04]]))


def chosen(n=4000, **kw) -> set[int]:
    g = torch.Generator().manual_seed(0)
    return {int(sample_next(LOGITS, generator=g, **kw)) for _ in range(n)}


def test_greedy():
    assert int(sample_next(LOGITS, temperature=0)) == 0


def test_plain_sampling_reaches_every_token():
    assert chosen() == {0, 1, 2, 3, 4}


def test_top_k():
    assert chosen(top_k=2) == {0, 1}


def test_top_p_keeps_the_smallest_set_reaching_p():
    assert chosen(top_p=0.45) == {0}            # 0.50 alone reaches 0.45
    assert chosen(top_p=0.6) == {0, 1}          # needs 0.50 + 0.25
    assert chosen(top_p=0.88) == {0, 1, 2}      # 0.90 >= 0.88


def test_min_p_keeps_tokens_close_to_the_best():
    assert chosen(min_p=0.25) == {0, 1, 2}      # >= 0.25 x 0.50 = 0.125
    assert chosen(min_p=0.6) == {0}


def test_low_temperature_concentrates():
    g = torch.Generator().manual_seed(0)
    picks = [int(sample_next(LOGITS, temperature=0.1, generator=g)) for _ in range(500)]
    assert picks.count(0) > 490


def test_repetition_penalty_pushes_seen_tokens_down():
    history = torch.tensor([[0]])
    assert int(sample_next(LOGITS, temperature=0, repetition_penalty=1.0, history=history)) == 0
    # log(0.5) is negative: the penalty multiplies it, making token 0 less likely than token 1
    assert int(sample_next(LOGITS, temperature=0, repetition_penalty=3.0, history=history)) == 1


def test_batches_are_independent():
    logits = torch.stack([LOGITS[0], LOGITS[0].flip(0)])
    assert sample_next(logits, temperature=0).flatten().tolist() == [0, 4]
