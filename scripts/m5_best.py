"""Print the run with the best validation exact match among <runs dir>/<prefix>*, and its learning rate.

    python scripts/m5_best.py lora-lr                      # runs/m5          -> "lora-lr3e-3 0.003"
    python scripts/m5_best.py lora-lr runs/m5-annotated

Used by run_m5.sh (and report_m5.py) to pick each method's learning rate on validation data only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def best_run(prefix: str, runs_dir: str = "runs/m5") -> tuple[str, dict]:
    rows = [(p.parent.name, json.loads(p.read_text())) for p in sorted(Path(runs_dir).glob(f"{prefix}*/results.json"))]
    if not rows:
        raise SystemExit(f"no finished runs match {runs_dir}/{prefix}*")
    return max(rows, key=lambda r: r[1]["val"]["exact_match"])


if __name__ == "__main__":
    name, res = best_run(sys.argv[1], *sys.argv[2:3])
    print(name, res["config"]["lr"])
