#!/usr/bin/env bash
# M4: generation speed and decoding strategies with the best model, then M3's open question —
# does RoPE alone explain the Llama recipe's win?
#
#     caffeinate -i bash scripts/run_m4.sh
#
# Safe to re-run: a finished training run is skipped and an interrupted one resumes.
set -euo pipefail

python scripts/bench_generate.py                  # -> docs/results/m4_generation.md, m4_latency.png (~3 min)
python scripts/decoding.py                        # -> docs/results/m4_decoding.md (~4 min)

# GPT-2 recipe with only the positions changed to RoPE (~50 min)
if [ -f checkpoints/gpt-rope/results.json ]; then
  echo "== gpt-rope: already trained, skipping"
elif [ -f checkpoints/gpt-rope/last.pt ]; then
  python scripts/train.py --resume --run gpt-rope
else
  python scripts/train.py --model gpt --run gpt-rope --pos rope
fi

python scripts/report.py --name m4_rope --runs gpt gpt-rope gpt-llama
python scripts/compare_models.py --ckpt checkpoints/gpt-rope/best.pt --ckpt checkpoints/gpt/best.pt \
    --ckpt checkpoints/gpt-llama/best.pt --out docs/results/m4_rope_compare.md --plot docs/results/m4_rope_context.png
python scripts/heads.py --ckpt checkpoints/gpt/best.pt --ckpt checkpoints/gpt-rope/best.pt \
    --ckpt checkpoints/gpt-llama/best.pt
python scripts/sample.py --stream --n 1 --top-p 0.9 --top-k 0
