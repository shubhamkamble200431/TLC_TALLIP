#!/usr/bin/env python
"""
Free disk space: for every COMPLETED run under experiments/, delete the
intermediate checkpoint-<step> directories (only needed to resume interrupted
training) and keep checkpoint-best/ (needed for eval/inference) untouched.

Triggered by an ENOSPC ("No space left on device") failure that killed
several in-flight runs on 2026-09-22 -- the whole filesystem was at 96%
(76 GiB free of 1.8 TiB) with 358 GiB sitting in stale intermediate
checkpoints from already-completed, already-evaluated runs.

    python scripts/prune_checkpoints.py            # do it
    python scripts/prune_checkpoints.py --dry-run  # just report
"""
import argparse
import json
import os
import shutil
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def du(path: str) -> int:
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=60).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "experiments"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    freed = 0
    n_dirs = 0
    n_runs = 0
    for dirpath, dirnames, _ in os.walk(args.root):
        if os.path.basename(dirpath) != "checkpoint":
            continue
        run_dir = os.path.dirname(dirpath)
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.isfile(status_path):
            continue
        try:
            status = json.load(open(status_path))
        except Exception:
            continue
        if status.get("status") != "completed":
            continue
        targets = [d for d in dirnames if d.startswith("checkpoint-")]
        dirnames[:] = [d for d in dirnames if not d.startswith("checkpoint-")]  # don't descend into what we're deleting
        if not targets:
            continue
        n_runs += 1
        for d in targets:
            p = os.path.join(dirpath, d)
            sz = du(p)
            freed += sz
            n_dirs += 1
            tag = "[dry-run] would delete" if args.dry_run else "deleting"
            print(f"  {tag} {p}  ({sz / 2**30:.2f} GiB)")
            if not args.dry_run:
                shutil.rmtree(p, ignore_errors=True)

    print(f"\n{'Would free' if args.dry_run else 'Freed'} {freed / 2**30:.1f} GiB "
          f"({n_dirs} checkpoint dirs across {n_runs} completed runs).")


if __name__ == "__main__":
    main()
