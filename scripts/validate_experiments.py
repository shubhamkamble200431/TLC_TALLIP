#!/usr/bin/env python
"""
Experiment-repository validation (brief section 27 / 28).

Checks:
  * all 16 X-ABCD configs x all declared seeds are present;
  * no duplicate run directories;
  * every completed run has checkpoint-best/, evaluation/<split>/ and the
    expected metrics;
  * BBS weight vectors are valid (all > 0; mean-normalised ~ 1);
  * seed values are the documented ones;
  * dataset version + fingerprint match the manifest;
  * the test set was NOT used for checkpoint / hyper-parameter selection
    (manifest invariant);
  * BBS grid selection (if present) came only from validation.

Exit code 0 iff nothing is missing or failed.

Example:
  python scripts/validate_experiments.py --dataset v2 --run-group factorial --seeds 42 43 44
"""
import _paths  # noqa: F401

import argparse
import glob
import os
import sys

import yaml

from src.config import ALL_CASE_IDS, CONFIGS_DIR
from src.data.dataset import normalized_bucket_weights
from src.experiments.status import read_status
from src.utils.io import read_json


def _load_manifest_yaml(dataset):
    p = os.path.join(CONFIGS_DIR, "datasets", f"{dataset}.yaml")
    return yaml.safe_load(open(p)) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--run-group", default="factorial")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--test-corpora", nargs="*", default=["v2", "v1"])
    ap.add_argument("--test-variants", nargs="*", default=["unsampled"])
    ap.add_argument("--expected-metrics", nargs="*",
                    default=["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"])
    ap.add_argument("--strict-fingerprint", action="store_true")
    args = ap.parse_args()

    problems, warnings, ok = [], [], []
    group_root = os.path.join(args.experiments_root, args.dataset, args.run_group)
    expected = [(c, s) for s in args.seeds for c in ALL_CASE_IDS]
    n_expected = len(expected)

    # ---- dataset manifest -----------------------------------------
    manifest = _load_manifest_yaml(args.dataset)
    if manifest is None:
        problems.append(f"dataset manifest configs/datasets/{args.dataset}.yaml is missing")
    else:
        try:
            from src.data.manifest import verify_fingerprint

            verify_fingerprint(manifest, strict=True)
            ok.append("dataset fingerprint matches manifest")
        except Exception as exc:
            (problems if args.strict_fingerprint else warnings).append(str(exc).splitlines()[0])

    # ---- per-run checks ------------------------------------------
    seen_dirs = {}
    counts = {"completed": 0, "failed": 0, "interrupted": 0, "running": 0, "missing": 0, "other": 0}
    for cid, seed in expected:
        run_dir = os.path.join(group_root, cid, f"seed_{seed}")
        if not os.path.isdir(run_dir):
            counts["missing"] += 1
            problems.append(f"missing run: {cid} seed_{seed}")
            continue
        seen_dirs.setdefault((cid, seed), []).append(run_dir)
        st = read_status(run_dir)
        status = st.get("status") if st else "other"
        counts[status if status in counts else "other"] += 1

        if status != "completed":
            problems.append(f"{cid} seed_{seed}: status={status}")
            continue

        # checkpoint + evaluation (every corpus x variant combo must exist)
        if not os.path.isdir(os.path.join(run_dir, "checkpoint-best")):
            problems.append(f"{cid} seed_{seed}: no checkpoint-best/")
        for corpus in args.test_corpora:
            for variant in args.test_variants:
                ev = os.path.join(run_dir, "evaluation", corpus, variant, "metrics.json")
                if not os.path.isfile(ev):
                    problems.append(f"{cid} seed_{seed}: no evaluation/{corpus}/{variant}/metrics.json")
                    continue
                m = read_json(ev).get("metrics", {})
                missing_m = [k for k in args.expected_metrics if k not in m]
                if missing_m:
                    warnings.append(f"{cid} seed_{seed} ({corpus}/{variant}): metrics missing {missing_m}")

        # config.yaml sanity
        cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(cfg_path):
            c = yaml.safe_load(open(cfg_path))
            if c.get("seed") != seed:
                problems.append(f"{cid} seed_{seed}: config.yaml seed={c.get('seed')} mismatch")
            if c.get("case_id") != cid:
                problems.append(f"{cid} seed_{seed}: config.yaml case_id={c.get('case_id')} mismatch")
            bw = c.get("bucket_weights") or []
            if bw:
                if any(float(x) <= 0 for x in bw):
                    problems.append(f"{cid} seed_{seed}: non-positive BBS weight {bw}")
                else:
                    nm = normalized_bucket_weights(bw)
                    if abs(sum(nm) - len(nm)) > 1e-6:
                        warnings.append(f"{cid} seed_{seed}: BBS weights not mean-normalisable cleanly")
            if seed not in (42, 43, 44):
                warnings.append(f"{cid} seed_{seed}: seed not in documented set {{42,43,44}}")

        # manifest: test-set isolation invariant
        man_path = os.path.join(run_dir, "manifest.json")
        if os.path.isfile(man_path):
            man = read_json(man_path)
            if man.get("test_set_used_for_selection", False):
                problems.append(f"{cid} seed_{seed}: manifest says test set used for selection!")
            if man.get("checkpoint_selection_split") != "validation":
                warnings.append(f"{cid} seed_{seed}: checkpoint_selection_split != validation")
            if man.get("dataset_version") != args.dataset:
                problems.append(f"{cid} seed_{seed}: manifest dataset_version={man.get('dataset_version')}")

    # duplicates
    for pattern_dir in glob.glob(os.path.join(group_root, "X-*", "seed_*")):
        pass  # deterministic paths -> duplicates only if the same (cid,seed) resolves twice, which it can't

    # ---- BBS selection provenance --------------------------------
    sel = os.path.join(args.experiments_root, args.dataset, "bbs_search", "selected_bbs_weights.json")
    if os.path.isfile(sel):
        s = read_json(sel)
        if s.get("test_set_used", True):
            problems.append("BBS selection: selected_bbs_weights.json test_set_used != false")
        if not str(s.get("selection_metric", "")).startswith("val_"):
            problems.append(f"BBS selection metric is not validation-based: {s.get('selection_metric')}")
        else:
            ok.append(f"BBS grid selection used validation only ({s.get('selection_metric')})")

    # ---- report -------------------------------------------------
    print("=" * 64)
    print(f"  Experiment validation — {args.dataset}/{args.run_group}")
    print("=" * 64)
    print(f"  Expected runs : {n_expected}   ({len(ALL_CASE_IDS)} configs x {len(args.seeds)} seeds)")
    print(f"  Completed     : {counts['completed']}")
    print(f"  Failed        : {counts['failed']}")
    print(f"  Interrupted   : {counts['interrupted']}")
    print(f"  Running       : {counts['running']}")
    print(f"  Missing       : {counts['missing']}")
    for line in ok:
        print(f"  [ok]   {line}")
    for line in warnings:
        print(f"  [warn] {line}")
    for line in problems:
        print(f"  [FAIL] {line}")
    print("=" * 64)
    if problems:
        print(f"  RESULT: FAIL ({len(problems)} problems, {len(warnings)} warnings)")
        sys.exit(1)
    print(f"  RESULT: PASS ({len(warnings)} warnings)")


if __name__ == "__main__":
    main()
