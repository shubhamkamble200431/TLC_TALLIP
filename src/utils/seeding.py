"""
Strict, centralised random-seed control for TLC_TALLIP.

Every experiment funnels through `set_global_seed()` so that Python `random`,
NumPy, PyTorch (CPU + CUDA), cuDNN, the HF `transformers` helper, DataLoader
workers, and the bucket-balanced sampler all derive from a single integer.

Section 4 / 28 of the revision brief: "All random seeds must be explicitly
controlled" and "Make sure BBS sampling itself is deterministic given the seed".
"""
from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Seeds used by the revision experiments.  Documented here as the single source
# of truth; runners default to these unless the caller overrides them.
DEFAULT_SEEDS = (42, 43, 44)


@dataclass
class SeedReport:
    """What actually got seeded — persisted into every experiment manifest."""

    seed: int
    python_hashseed: Optional[str] = None
    torch_deterministic: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    cublas_workspace_config: Optional[str] = None
    nondeterministic_ops: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "python_hashseed": self.python_hashseed,
            "torch_deterministic": self.torch_deterministic,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cublas_workspace_config": self.cublas_workspace_config,
            "nondeterministic_ops": self.nondeterministic_ops,
        }


def set_global_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """
    Seed every practical RNG source.

    Parameters
    ----------
    seed
        The integer seed (42 / 43 / 44 for the main experiments).
    deterministic
        If True, also request deterministic cuDNN / CUDA kernels.  Some ops
        (e.g. certain scatter/atomic kernels) have no deterministic
        implementation; we fall back to `warn_only=True` so training does not
        crash, and record the fact in the returned SeedReport.

    Returns
    -------
    SeedReport
        A machine-readable description of what was configured, for the manifest.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    report = SeedReport(seed=seed, python_hashseed=str(seed))

    # HF convenience FIRST — it also lazily imports TensorFlow on its first call,
    # which perturbs the `random` module; we re-seed everything explicitly below
    # so the final RNG state is deterministic regardless of that.
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except Exception:
        pass

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception as exc:  # pragma: no cover - numpy always present in practice
        logger.warning("Could not seed NumPy: %s", exc)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # cuBLAS needs this env var *before* the first CUDA call for
            # deterministic matmul; we set it defensively regardless.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            report.cublas_workspace_config = os.environ["CUBLAS_WORKSPACE_CONFIG"]

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            report.cudnn_deterministic = True
            report.cudnn_benchmark = False

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                report.torch_deterministic = True
            except Exception as exc:  # older torch
                logger.warning("torch.use_deterministic_algorithms unavailable: %s", exc)
        else:
            torch.backends.cudnn.benchmark = True
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not seed PyTorch: %s", exc)

    logger.info(
        "[seed] global seed=%d deterministic=%s (cudnn.deterministic=%s, "
        "torch.use_deterministic_algorithms=%s)",
        seed,
        deterministic,
        report.cudnn_deterministic,
        report.torch_deterministic,
    )
    return report


def seed_worker(worker_id: int) -> None:
    """
    `worker_init_fn` for DataLoader workers.

    PyTorch seeds `torch` per-worker for us, but NumPy / `random` inside a
    worker inherit the parent process state unless we reseed them explicitly.
    """
    try:
        import torch

        base = torch.initial_seed() % 2 ** 32
    except Exception:
        base = worker_id
    random.seed(base)
    try:
        import numpy as np

        np.random.seed(base % 2 ** 32)
    except Exception:
        pass


def make_generator(seed: int):
    """A dedicated `torch.Generator` for samplers / shuffling (never the global RNG)."""
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
