#!/usr/bin/env bash
# M3: train the three GPTs, then compare everything with M2's baselines.
#
#     caffeinate -i bash scripts/run_m3.sh        # caffeinate keeps the Mac awake while it runs
#
# Safe to re-run: finished runs are skipped and an interrupted run resumes from its last checkpoint.
set -euo pipefail

train() {                         # train <run name> [model flags...]
  local name=$1; shift
  if [ -f "checkpoints/$name/results.json" ]; then
    echo "== $name: already trained, skipping"
  elif [ -f "checkpoints/$name/last.pt" ]; then
    echo "== $name: resuming"
    python scripts/train.py --resume --run "$name"
  else
    echo "== $name"
    python scripts/train.py --model gpt --run "$name" "$@"
  fi
}

train gpt                                                  # GPT-2 recipe, 6 x 384, 13.9M parameters
train gpt-small --d-model 256 --n-layers 4 --n-heads 4    # GPT-2 recipe, 4 x 256, 5.3M (= the LSTM's size)
train gpt-llama --style llama                              # Llama recipe, 6 x 384, 13.8M

python scripts/report.py --name m3 --runs lstm gpt-small gpt gpt-llama
python scripts/compare_models.py --ckpt checkpoints/gpt-llama/best.pt --ckpt checkpoints/gpt/best.pt \
    --ckpt checkpoints/gpt-small/best.pt --ckpt checkpoints/lstm/best.pt \
    --out docs/results/m3_compare.md --plot docs/results/m3_context.png
python scripts/attention.py --ckpt checkpoints/gpt/best.pt
python scripts/attention.py --ckpt checkpoints/gpt-llama/best.pt --out docs/results/attention_llama.png
python scripts/sample.py --ckpt checkpoints/gpt-llama/best.pt
