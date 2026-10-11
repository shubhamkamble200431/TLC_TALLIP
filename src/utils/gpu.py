"""GPU discovery + assignment helpers for the job generator (section 34)."""
from __future__ import annotations

import os
import subprocess
from typing import List


def detect_gpus() -> List[int]:
    """Return visible GPU indices via PyTorch; fall back to nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return [int(x) for x in out.stdout.split()]
    except Exception:
        pass
    return []


def gpu_free_memory_mib() -> dict:
    """index -> free MiB, via nvidia-smi (best effort)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            res = {}
            for line in out.stdout.strip().splitlines():
                idx, free = [p.strip() for p in line.split(",")]
                res[int(idx)] = int(free)
            return res
    except Exception:
        pass
    return {}


def round_robin_assign(job_ids: List[str], gpus: List[int]) -> List[tuple]:
    """
    Assign jobs to GPUs round-robin.  One large training job per GPU at a time
    (section 34: "Avoid assigning multiple large training jobs to the same GPU").
    Returns list of (job_id, gpu_index).
    """
    if not gpus:
        return [(j, None) for j in job_ids]
    return [(j, gpus[i % len(gpus)]) for i, j in enumerate(job_ids)]
