#!/usr/bin/env bash
# M6: package the two models, test the service, and measure it under load (~10 minutes, ~17 with Node.js).
#
#     bash scripts/run_m6.sh
#
# 1. models/lm and models/parser: safetensors bundles + model card, checked against the checkpoints
#    and — on the whole validation set — against M5's own decoding
# 2. every test, the new serving ones included (the browser port's tests need Node.js)
# 3. what the language model knows: docs/results/m6_facts.md (20 capitals; ~1 minute)
# 4. load test: docs/results/m6_load.md, m6_load.json, m6_parse.png
# 5. with Node.js: the browser version against PyTorch, docs/results/m6_web.md (~7 minutes)
#
# Docker and the Hugging Face upload are separate steps (README, "M6 — ship").
set -euo pipefail

python scripts/export.py --verify 1390
pytest -q
python scripts/probe_facts.py
python scripts/load_test.py
if command -v node > /dev/null; then
    python scripts/check_web.py
else
    echo "Node.js not found: skipping scripts/check_web.py (brew install node)"
fi
