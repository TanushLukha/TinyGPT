import pytest
import torch
import torch.nn as nn

from tinygpt.lora import LoRALinear, apply_lora, count_params, load_lora, lora_layers, lora_state_dict, merge_lora
from tinygpt.models import GPT, GPTConfig

STYLES = ["gpt2", "llama"]


def tiny(style="gpt2", seed=0, **kw) -> GPT:
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0, style=style)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg))


def adapted(style="gpt2", r=4, targets=("attn", "mlp")):
    base = tiny(style).eval()
    model = tiny(style).eval()
    model.load_state_dict(base.state_dict())
    apply_lora(model, r=r, alpha=8, targets=targets)
    return base, model


def train_a_little(model, steps=5, lr=1e-2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    idx = torch.randint(0, 50, (4, 16), generator=torch.Generator().manual_seed(1))
    for _ in range(steps):
        _, loss = model(idx, idx.roll(-1, dims=1))
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_lora_linear_is_w_plus_scaled_ba():
    """The layer computes x (W + (α/r)·BA)ᵀ + b."""
    torch.manual_seed(0)
    base = nn.Linear(6, 5)
    lin = LoRALinear(base, r=3, alpha=6)
    with torch.no_grad():
        lin.lora_B.normal_()
    x = torch.randn(4, 6)
    expected = x @ (base.weight + 2.0 * lin.lora_B @ lin.lora_A).T + base.bias
    torch.testing.assert_close(lin(x), expected)


@pytest.mark.parametrize("style", STYLES)
def test_starts_exactly_as_the_pretrained_model(style):
    """B = 0 at the start, so the adapted model computes exactly what the base model computes."""
    base, model = adapted(style)
    idx = torch.randint(0, 50, (2, 16))
    assert torch.equal(model(idx)[0], base(idx)[0])


@pytest.mark.parametrize("style,targets,expected", [
    ("gpt2", ("attn",), ["attn.qkv", "attn.proj"]),
    ("gpt2", ("attn", "mlp"), ["attn.qkv", "attn.proj", "mlp.fc", "mlp.proj"]),
    ("llama", ("mlp",), ["mlp.gate", "mlp.up", "mlp.proj"]),
])
def test_targets_and_trainable_parameter_count(style, targets, expected):
    base, model = adapted(style, r=4, targets=targets)
    names = list(lora_layers(model))
    assert names == [f"blocks.{i}.{s}" for i in range(2) for s in expected]
    trainable, total = count_params(model)
    assert trainable == sum(4 * (m.base.in_features + m.base.out_features) for m in lora_layers(model).values())
    assert total == count_params(base)[1] + trainable
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".lora_" not in n)


def test_unknown_target_is_an_error():
    with pytest.raises(ValueError):
        apply_lora(tiny(), targets=("embeddings",))


@pytest.mark.parametrize("style", STYLES)
def test_training_changes_only_the_adapter(style):
    base, model = adapted(style)
    frozen = {k: v.clone() for k, v in model.state_dict().items() if ".lora_" not in k}
    train_a_little(model)
    for k, v in model.state_dict().items():
        if ".lora_" not in k:
            assert torch.equal(v, frozen[k]), k
    assert any(m.lora_B.abs().sum() > 0 for m in lora_layers(model).values())
    idx = torch.randint(0, 50, (2, 16))
    assert not torch.allclose(model(idx)[0], base(idx)[0])


@pytest.mark.parametrize("style", STYLES)
def test_merging_gives_the_same_model(style):
    """W ← W + ΔW, unwrapped: same outputs, original parameter names, no LoRA layers left."""
    base, model = adapted(style)
    train_a_little(model)
    idx = torch.randint(0, 50, (2, 16))
    before = model(idx)[0]
    merge_lora(model)
    assert not lora_layers(model)
    assert list(model.state_dict()) == list(base.state_dict())
    torch.testing.assert_close(model(idx)[0], before, atol=1e-5, rtol=1e-4)
    fresh = tiny(style).eval()
    fresh.load_state_dict(model.state_dict())                  # loads into a plain GPT
    torch.testing.assert_close(fresh(idx)[0], before, atol=1e-5, rtol=1e-4)


def test_adapter_round_trip():
    _, model = adapted()
    train_a_little(model)
    state = lora_state_dict(model)
    assert all(".lora_" in k for k in state)
    _, other = adapted()
    load_lora(other, state)
    idx = torch.randint(0, 50, (2, 16))
    torch.testing.assert_close(other.eval()(idx)[0], model.eval()(idx)[0])
    _, attn_only = adapted(targets=("attn",))
    with pytest.raises(KeyError):
        load_lora(attn_only, state)


def test_kv_cache_works_with_adapters():
    _, model = adapted("llama")
    train_a_little(model)
    model.eval()
    idx = torch.randint(0, 50, (2, 16))
    full, _ = model(idx)
    cache = model.new_cache(batch=2)
    parts = [model(idx[:, :5], cache=cache)[0]] + [model(idx[:, t:t + 1], cache=cache)[0] for t in range(5, 16)]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-5, rtol=1e-4)
