#!/usr/bin/env python
"""
Build (or refresh) a dataset manifest from an EXISTING on-disk corpus
(brief section 2: dataset version control).

There is no augmentation step: the two Sanskrit->Hindi corpora are used exactly
as pasted into `data/`:

    data/v2/   -> the MAIN corpus for the 16 x 3 NLLB + 16 x 3 mBART-50 factorials
                  (train/val + raw `test.sa` held-out test)
    data/v1/   -> the older/shorter corpus, used ONLY by the V1 + a*V2
                  data-scale ablation

This script computes per-split line counts, source-length bucket counts and a
SHA-256 fingerprint, and writes `configs/datasets/<version>.yaml`.  The training
pipeline verifies that fingerprint before every run.

Usage:
    python scripts/prepare_dataset.py --version v2                 # main corpus
    python scripts/prepare_dataset.py --version v1 --test-split test_5414_5
"""
import _paths  # noqa: F401

import argparse
import os

import yaml

from src.config import CONFIGS_DIR, REPO_ROOT
from src.data.manifest import build_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=["v1", "v2"],
                    help="dataset version / data dir name under data/")
    ap.add_argument("--data-dir", default=None, help="override (default: data/<version>)")
    ap.add_argument("--test-split", default=None,
                    help="test file prefix (default: test_7264_5 for v2, test_5414_5 for v1)")
    ap.add_argument("--src-ext", default="sa")
    ap.add_argument("--tgt-ext", default="hi")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(REPO_ROOT, "data", args.version)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"{data_dir} does not exist")
    test_split = args.test_split or ("test_7264_5" if args.version == "v2" else "test_5414_5")
    s, t = args.src_ext, args.tgt_ext

    splits = {
        "train": {"src": f"train.{s}", "tgt": f"train.{t}"},
        "val": {"src": f"val.{s}", "tgt": f"val.{t}"},
        "test": {"src": f"{test_split}.{s}", "tgt": f"{test_split}.{t}"},
    }
    if test_split == "test":
        # no bucket-balanced ("sampled") variant -- evaluate on the raw test
        # set only (test.sa / test.hi); results land in evaluation/<corpus>/unsampled/
        test_variants = {
            "unsampled": {"src": f"test.{s}", "tgt": f"test.{t}"},
        }
    else:
        # every model is evaluated on BOTH the bucket-balanced ("sampled") and
        # the raw ("unsampled") test set -> results land in evaluation/<corpus>/sampled|unsampled/
        test_variants = {
            "sampled": {"src": f"{test_split}.{s}", "tgt": f"{test_split}.{t}"},
            "unsampled": {"src": f"test.{s}", "tgt": f"test.{t}"},
        }
    for name, files in {**splits, **{f"test:{k}": v for k, v in test_variants.items()}}.items():
        for f in files.values():
            p = os.path.join(data_dir, f)
            if not os.path.isfile(p):
                raise SystemExit(f"missing {name} file: {p}")

    manifest = build_manifest(
        version=args.version,
        data_dir=data_dir,
        splits=splits,
        src_lang="san_Deva",
        tgt_lang="hin_Deva",
        preprocessing={
            "version": "as-pasted",
            "augmentation": "none",
            "transliteration": "none",
            "notes": f"corpus used verbatim from data/{args.version}/; no synthetic data",
        },
        tokenizer="facebook/nllb-200-distilled-600M + facebook/mbart-large-50-many-to-many-mmt",
        test_variants=test_variants,
    )
    manifest["paths"]["data_dir"] = data_dir
    manifest["role"] = ("main factorial + mBART baseline + BBS/TLC search"
                        if args.version == "v2" else "V1+a*V2 data-scale ablation only")

    out_yaml = os.path.join(CONFIGS_DIR, "datasets", f"{args.version}.yaml")
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    st = manifest["statistics"]
    print(f"[prepare] version={args.version}  data_dir={data_dir}")
    print(f"  n_train={st['n_train']:,}  n_val={st['n_val']:,}")
    variant_str = "  ".join(f"{k}={v:,}" for k, v in st["n_test_variants"].items())
    print(f"  test variants: {variant_str}")
    print(f"  fingerprint={manifest['fingerprint']}")
    for split, sd in st["per_split"].items():
        print(f"  {split:<16} source buckets B1..B5: {sd['source_bucket_counts']}")
    print(f"  -> {out_yaml}")


if __name__ == "__main__":
    main()
