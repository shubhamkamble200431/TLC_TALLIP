"""
Expansion of the main 3-seed 2^4 factorial (brief sections 1, 3, 29-A).

    dataset = v2        (ONLY — never multiple dataset versions)
    seeds   = [42, 43, 44]
    configs = 16 canonical X-ABCD  (A=BBS B=TLC C=RoPE D=ALiBi)
    => 48 runs
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from src.config import ALL_CASE_IDS, parse_case_id
from src.utils.seeding import DEFAULT_SEEDS


def resolve_configs(spec) -> List[str]:
    if spec in (None, "all", ["all"]):
        return list(ALL_CASE_IDS)
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for c in spec:
        cid = c.strip().upper()
        parse_case_id(cid)  # validates
        out.append(cid)
    return out


def resolve_seeds(spec) -> List[int]:
    if spec in (None, "default"):
        return list(DEFAULT_SEEDS)
    return [int(s) for s in spec]


def enumerate_runs(configs=None, seeds=None) -> List[Tuple[str, int]]:
    """(case_id, seed) pairs, seed-major to match `for seed: for config:` in the brief."""
    cfgs = resolve_configs(configs)
    sds = resolve_seeds(seeds)
    return [(cid, s) for s in sds for cid in cfgs]
