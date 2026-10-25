#!/usr/bin/env bash
# M5: fine-tune TinyGPT to parse Hinglish assistant requests — LoRA vs full fine-tuning vs training
# from scratch vs no training at all.
#
#     caffeinate -i bash scripts/run_m5.sh        # safe to re-run: finished runs are skipped
#
# 1. a clean base model: gpt-llama retrained without the validation/test requests (~55 min)
# 2. what the model writes, and each method's learning rate — chosen on validation data only
# 3. LoRA rank, layers, label format, seeds, longer training; data efficiency; contamination
# 4. tables and charts -> docs/results/m5_*.md, *.png
#
# M5_FLAGS adds flags to every fine-tuning run, e.g. a quick check of the whole pipeline:
#     M5_FLAGS="--steps 4 --val-size 20" bash scripts/run_m5.sh
set -euo pipefail

# ---- 1. clean base model -------------------------------------------------------------------
if [ ! -f data/tokens_dc/train.bin ]; then
  python scripts/decontaminate.py
  python scripts/encode_dataset.py --data data/clean_dc --out data/tokens_dc
fi
if [ -f checkpoints/gpt-llama-clean/results.json ]; then
  echo "== gpt-llama-clean: already trained, skipping"
elif [ -f checkpoints/gpt-llama-clean/last.pt ]; then
  python scripts/train.py --resume --run gpt-llama-clean
else
  python scripts/train.py --model gpt --style llama --run gpt-llama-clean --data-dir data/tokens_dc
fi

run() {                           # run <dir> <name> [finetune.py flags...]: train once, skip if done
  local dir=$1 name=$2; shift 2
  if [ -f "$dir/$name/results.json" ]; then
    echo "== $dir/$name: done, skipping"
  else
    echo "== $dir/$name"
    python scripts/finetune.py --out-dir "$dir" --run "$name" --eval-every 150 "$@" ${M5_FLAGS:-}
  fi
}
test_eval() {                     # test_eval <dir> <name>: score a finished run on the test set, once
  if ! grep -q '"test": {' "$1/$2/results.json"; then
    python scripts/finetune.py --out-dir "$1" --run "$2" --eval-only --test
  fi
}
val_em() { python -c "import json, sys; print(json.load(open(sys.argv[1]))['val']['exact_match'])" "$1/$2/results.json"; }

# ---- 2. what the model writes: the decoupled tree, or the request annotated with brackets ----
run runs/m5 lora-lr3e-3 --method lora --lr 3e-3
run runs/m5-annotated lora-lr3e-3 --method lora --lr 3e-3 --target annotated
if python -c "import sys; sys.exit(float(sys.argv[1]) <= float(sys.argv[2]))" \
     "$(val_em runs/m5-annotated lora-lr3e-3)" "$(val_em runs/m5 lora-lr3e-3)"; then
  TARGET=annotated; RUNS=runs/m5-annotated; OTHER=runs/m5
else
  TARGET=decoupled; RUNS=runs/m5; OTHER=runs/m5-annotated
fi
echo "== output chosen on validation: $TARGET"
ft() { local name=$1; shift; run "$RUNS" "$name" --target "$TARGET" "$@"; }

# ---- learning rates (validation only) -----------------------------------------------------------
for lr in 1e-3 3e-3 1e-2; do ft "lora-lr$lr" --method lora --lr "$lr"; done
for lr in 1e-4 3e-4 1e-3; do ft "full-lr$lr" --method full --lr "$lr"; done
for lr in 3e-4 1e-3 3e-3; do ft "scratch-lr$lr" --method scratch --lr "$lr"; done
read -r LORA LORA_LR < <(python scripts/m5_best.py lora-lr "$RUNS")
read -r FULL FULL_LR < <(python scripts/m5_best.py full-lr "$RUNS")
read -r SCRATCH SCRATCH_LR < <(python scripts/m5_best.py scratch-lr "$RUNS")
echo "== chosen on validation: LoRA $LORA_LR, full $FULL_LR, scratch $SCRATCH_LR"
for r in "$LORA" "$FULL" "$SCRATCH"; do test_eval "$RUNS" "$r"; done
test_eval "$OTHER" lora-lr3e-3                       # the other output format, for comparison
ft fewshot-k3 --method fewshot --test

# ---- 3. what matters for LoRA, how much data, and contamination -------------------------------
for r in 1 4 16 64; do ft "lora-r$r" --method lora --lr "$LORA_LR" --rank "$r" --test; done
ft lora-attn --method lora --lr "$LORA_LR" --targets attn --test
ft lora-upper --method lora --lr "$LORA_LR" --labels upper --test
for s in 1 2; do ft "lora-seed$s" --method lora --lr "$LORA_LR" --seed "$s" --test; done
ft lora-long --method lora --lr "$LORA_LR" --steps 1800 --eval-every 450 --test
ft full-long --method full --lr "$FULL_LR" --steps 1800 --eval-every 450 --test
for n in 100 1000; do
  ft "lora-n$n" --method lora --lr "$LORA_LR" --train-size "$n" --test
  ft "full-n$n" --method full --lr "$FULL_LR" --train-size "$n" --test
  ft "scratch-n$n" --method scratch --lr "$SCRATCH_LR" --train-size "$n" --test
done
ft orig-lora --method lora --lr "$LORA_LR" --base checkpoints/gpt-llama/best.pt --test
ft orig-full --method full --lr "$FULL_LR" --base checkpoints/gpt-llama/best.pt --test

# ---- 4. report ---------------------------------------------------------------------------------
python scripts/report_m5.py --runs "$RUNS" --other-runs "$OTHER"
python scripts/parse.py --out-dir "$RUNS" --run "$LORA"
