import json

import numpy as np
import pytest


@pytest.fixture
def token_dir(tmp_path):
    """Factory for a fake data/tokens directory (train.bin, val.bin, meta.json).

    Each source is a run of random tokens ending in <|endoftext|>, and claims 3 bytes per token.
    """
    def make(sizes=None, V=64, seed=0, pattern=None):
        sizes = sizes or {"wiki_hi": 120, "findnitai": 60}
        rng = np.random.default_rng(seed)
        eot = V - 1
        meta = {"tokenizer": "none", "vocab_size": V, "eot_id": eot, "dtype": "uint16", "splits": {}}
        for split in ("train", "val"):
            parts, sources = [], {}
            for src, n in sizes.items():
                if pattern is not None:                       # a repeating sequence: easy to learn
                    toks = np.resize(np.asarray(pattern), n).astype(np.uint16)
                else:
                    toks = rng.integers(0, V - 1, n).astype(np.uint16)
                toks[-1] = eot
                parts.append(toks)
                sources[src] = {"docs": 1, "tokens": n, "bytes": 3 * n}
            np.concatenate(parts).tofile(tmp_path / f"{split}.bin")
            meta["splits"][split] = {"tokens": sum(sizes.values()), "sources": sources}
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        return tmp_path
    return make
