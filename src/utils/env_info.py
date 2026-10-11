"""Environment / provenance capture for the experiment manifest (section 25)."""
from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Optional


def _run(cmd: list) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    repo_dir = repo_dir or os.getcwd()
    return _run(["git", "-C", repo_dir, "rev-parse", "HEAD"])


def git_dirty(repo_dir: Optional[str] = None) -> Optional[bool]:
    repo_dir = repo_dir or os.getcwd()
    status = _run(["git", "-C", repo_dir, "status", "--porcelain"])
    if status is None:
        return None
    return bool(status.strip())


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def collect() -> dict:
    info = {
        "timestamp": now_iso(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info
