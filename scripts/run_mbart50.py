#!/usr/bin/env python
"""
mBART-50-M2M  FULL 2^4 factorial  (brief sections 5, 22, 29-B, 37-P2).

Same 16 X-ABCD configs x 3 seeds = 48 runs as the NLLB factorial, on data/v2 —
to check that the dataset-level findings GENERALISE to a second architecture.

This is a thin alias for the generic runner: it forces the mBART-50 experiment
profile (configs/experiments/mbart50.yaml -> model_key mbart50_many_to_many,
run_group "mbart50", configs: all) and otherwise behaves exactly like
scripts/run_factorial.py — same --bbs-weights, --resume, --skip-completed,
--dry-run, --list, smoke-test knobs, and held-out eval wiring.

    python scripts/run_mbart50.py --run-all                     # all 48
    python scripts/run_mbart50.py --config X-1101 --seed 42     # one cell
    python scripts/run_mbart50.py --run-all --dry-run           # plan only
    python scripts/run_mbart50.py --config X-0000 --seed 42 \\
        --epochs 1 --max-train-samples 200                      # smoke test

Equivalent to:  python scripts/run_factorial.py --experiment mbart50 <args...>
"""
import _paths  # noqa: F401

import sys

import run_factorial

# A bare `run_mbart50.py` (no run selector at all) means "the whole 16 x 3 grid".
# Any explicit selector (--run-all / --config / --configs / --seed / --seeds /
# --list / --dry-run alone) is passed straight through to run_factorial.
_RUN_SELECTORS = {"--run-all", "--config", "--configs", "--seed", "--seeds", "--list"}


def main():
    argv = list(sys.argv[1:])
    if "--experiment" not in argv:
        argv += ["--experiment", "mbart50"]
    if not (_RUN_SELECTORS & set(argv)):
        argv += ["--run-all"]
    sys.argv = [sys.argv[0], *argv]
    run_factorial.main()


if __name__ == "__main__":
    main()
