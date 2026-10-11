"""
Dataset fingerprinting + manifest (brief section 2: "Dataset version control").

A dataset version is identified by a stable fingerprint over the *content* of
its split files.  The training pipeline verifies this fingerprint before every
run and refuses to start if the data on disk has drifted from what the
`configs/datasets/<version>.yaml` manifest records — brief: "Do not silently
regenerate or alter the dataset during training."
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List

from src.data.buckets import DEFAULT_BOUNDARIES, bucket_counts
from src.utils.io import read_lines


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    return sum(1 for _ in open(path, "rb"))


def split_stats(data_dir: str, src_file: str, tgt_file: str,
                boundaries: List[int] = DEFAULT_BOUNDARIES) -> Dict:
    src_path = os.path.join(data_dir, src_file)
    tgt_path = os.path.join(data_dir, tgt_file)
    src_lines = read_lines(src_path)
    n_src = len(src_lines)
    n_tgt = _count_lines(tgt_path)
    return {
        "src_file": src_file,
        "tgt_file": tgt_file,
        "n_examples": n_src,
        "n_src_lines": n_src,
        "n_tgt_lines": n_tgt,
        "aligned": n_src == n_tgt,
        "src_sha256": sha256_file(src_path),
        "tgt_sha256": sha256_file(tgt_path),
        "source_bucket_counts": bucket_counts(src_lines, boundaries),
    }


def dataset_fingerprint(split_stat_map: Dict[str, Dict]) -> str:
    """Order-independent fingerprint over the per-split src/tgt SHA-256 digests."""
    h = hashlib.sha256()
    for split in sorted(split_stat_map):
        s = split_stat_map[split]
        h.update(split.encode())
        h.update(s["src_sha256"].encode())
        h.update(s["tgt_sha256"].encode())
    return h.hexdigest()


def build_manifest(
    version: str,
    data_dir: str,
    splits: Dict[str, Dict[str, str]],
    src_lang: str,
    tgt_lang: str,
    preprocessing: Dict,
    tokenizer: str,
    boundaries: List[int] = DEFAULT_BOUNDARIES,
    test_variants: Dict[str, Dict[str, str]] = None,
) -> Dict:
    """
    Parameters
    ----------
    splits
        ``{"train": {"src": "train.sa", "tgt": "train.hi"}, "val": {...},
           "test": {...}}``  -- 'test' is the SAMPLED (bucket-balanced) test set.
    test_variants
        ``{"sampled": {"src":..,"tgt":..}, "unsampled": {"src":"test.sa",
           "tgt":"test.hi"}}`` -- every model is evaluated on BOTH.  All named
        files are fingerprinted.
    """
    test_variants = test_variants or {
        "sampled": {"src": splits["test"]["src"], "tgt": splits["test"]["tgt"]}
    }
    fp_splits = dict(splits)
    for name, files in test_variants.items():
        fp_splits[f"test_{name}"] = files

    per_split = {
        name: split_stats(data_dir, files["src"], files["tgt"], boundaries)
        for name, files in fp_splits.items()
    }
    fp = dataset_fingerprint(per_split)
    return {
        "dataset_version": version,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "fingerprint": fp,
        "bucket_boundaries": boundaries,
        "preprocessing": preprocessing,
        "tokenizer": tokenizer,
        "paths": {
            "data_dir": data_dir,
            "train_src": splits["train"]["src"], "train_tgt": splits["train"]["tgt"],
            "val_src": splits["val"]["src"],     "val_tgt": splits["val"]["tgt"],
            "test_src": splits["test"]["src"],   "test_tgt": splits["test"]["tgt"],
        },
        "test_variants": test_variants,
        "statistics": {
            "n_train": per_split["train"]["n_examples"],
            "n_val": per_split["val"]["n_examples"],
            "n_test": per_split["test"]["n_examples"],
            "n_test_variants": {k: per_split[f"test_{k}"]["n_examples"] for k in test_variants},
            "per_split": per_split,
        },
    }


def verify_fingerprint(manifest: Dict, strict: bool = True) -> bool:
    """Recompute the fingerprint from files on disk and compare to the manifest."""
    data_dir = manifest["paths"]["data_dir"]
    boundaries = manifest.get("bucket_boundaries", DEFAULT_BOUNDARIES)
    per_split = {}
    for split in ("train", "val", "test"):
        per_split[split] = split_stats(
            data_dir,
            manifest["paths"][f"{split}_src"],
            manifest["paths"][f"{split}_tgt"],
            boundaries,
        )
    for name, files in (manifest.get("test_variants") or {}).items():
        per_split[f"test_{name}"] = split_stats(data_dir, files["src"], files["tgt"], boundaries)
    current = dataset_fingerprint(per_split)
    ok = current == manifest.get("fingerprint")
    if not ok and strict:
        raise RuntimeError(
            "Dataset fingerprint mismatch for version "
            f"{manifest.get('dataset_version')!r}:\n"
            f"  manifest : {manifest.get('fingerprint')}\n"
            f"  on disk  : {current}\n"
            "Refusing to train on drifted data.  Re-run scripts/prepare_dataset.py "
            "or point at the correct data_dir."
        )
    return ok
