"""Parse Hinglish assistant requests with a fine-tuned run from M5.

    python scripts/parse.py --run lora-lr1e-3                         # built-in examples
    python scripts/parse.py --run lora-lr1e-3 "kal subah 7 baje ka alarm laga do"

A LoRA run is loaded as base + adapter and merged (W + ΔW), so parsing costs exactly what the base
model costs. M6 serves a run after scripts/export.py has turned it into a bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from finetune import load_best  # noqa: E402

from tinygpt.bpe import BPETokenizer  # noqa: E402
from tinygpt.device import pick_device  # noqa: E402
from tinygpt.lora import merge_lora  # noqa: E402
from tinygpt.text import normalize  # noqa: E402
from tinygpt.top import from_model, parse_tree, predict, prompt_ids, scored_form, slot_items, top_intent  # noqa: E402

EXAMPLES = ["kal subah 7 baje ka alarm laga do", "mujhe Delhi ka weather batao",
            "agle hafte mummy ko call karne ka reminder set karo", "Arijit Singh ke gaane bajao",
            "office se ghar pahunchne mein kitna time lagega"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requests", nargs="*", help="default: a few built-in examples")
    ap.add_argument("--run", required=True, help="a run directory in runs/m5/")
    ap.add_argument("--out-dir", default="runs/m5")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run_dir = Path(args.out_dir) / args.run
    cfg = argparse.Namespace(**json.loads((run_dir / "config.json").read_text()))
    device = pick_device(args.device)
    model = merge_lora(load_best(cfg, run_dir, device)).eval()
    tok = BPETokenizer.load(cfg.tokenizer)

    requests = [normalize(r) for r in (args.requests or EXAMPLES)]
    t0 = time.perf_counter()
    raw = predict(model, tok, [prompt_ids(tok, r) for r in requests], device)
    ms = (time.perf_counter() - t0) * 1000 / len(requests)
    annotated = getattr(cfg, "target", "decoupled") == "annotated"
    for req, out in zip(requests, raw):
        written = from_model(out, cfg.labels)
        parse = scored_form(written) if annotated else written
        print(f"\n{req}")
        if annotated:
            print(f"  {written}")                  # what the model wrote: the request with brackets
        print(f"  {parse}")
        try:
            slots = ", ".join(f"{label[3:]} = {text!r}" for label, text in slot_items(parse_tree(parse)))
            print(f"  intent {top_intent(parse)[3:]}" + (f" | {slots}" if slots else ""))
        except ValueError:
            print("  (not a valid tree)")
    print(f"\n{ms:.0f} ms per request on {device.type}")


if __name__ == "__main__":
    main()
