"""
BBS weight-vector search space (brief sections 6, 7).

Rules
  * 5 buckets -> vector [w1..w5], wi > 0;
  * mean-normalise every candidate so mean(w) = 1  <=>  sum(w) = 5
    (never rescale the whole vector — that would change sampling intensity);
  * optional monotonicity  w1 <= w2 <= w3 <= w4 <= w5  (increasing emphasis on
    longer source sentences);
  * de-duplicate AFTER normalisation;
  * three search modes:
      exhaustive : cartesian product of `values` over all 5 buckets
                   (use only with a SMALL `values` list);
      coordinate : vary ONE bucket at a time around an anchor;
      grid2d     : full 2D sweep of two buckets (e.g. w4 x w5) with the other
                   three held at the anchor -> directly fills the heatmap.
  * always cap at `max_candidates`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

N_BUCKETS = 5


def normalize_mean(vec: List[float]) -> List[float]:
    v = [float(x) for x in vec]
    if any(x <= 0 for x in v):
        raise ValueError(f"all weights must be > 0, got {vec}")
    s = sum(v)
    return [x * N_BUCKETS / s for x in v]  # sum -> N_BUCKETS, mean -> 1


def is_monotonic(vec: List[float], tol: float = 1e-9) -> bool:
    return all(vec[i] <= vec[i + 1] + tol for i in range(len(vec) - 1))


def _key(vec: List[float], places: int = 4) -> Tuple:
    return tuple(round(x, places) for x in vec)


@dataclass
class Candidate:
    id: str
    raw: List[float]
    normalized: List[float]
    tag: str = ""


@dataclass
class SearchSpace:
    values: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    normalize_mean: bool = True
    monotonic: bool = True
    mode: str = "grid2d"
    anchor: List[float] = field(default_factory=lambda: [1.0] * N_BUCKETS)
    coordinate_axes: List[int] = field(default_factory=lambda: [3, 4])
    max_candidates: int = 40
    always_include: Dict[str, List[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _raw_vectors(self) -> List[Tuple[List[float], str]]:
        vecs: List[Tuple[List[float], str]] = []
        # always_include FIRST so those vectors claim their tag before an
        # identical grid point dedups them away.
        for tag, vec in self.always_include.items():
            vecs.append((list(vec), tag))
        if self.mode == "exhaustive":
            for combo in itertools.product(self.values, repeat=N_BUCKETS):
                vecs.append((list(combo), "exhaustive"))
        elif self.mode == "coordinate":
            for ax in self.coordinate_axes:
                for val in self.values:
                    v = list(self.anchor)
                    v[ax] = val
                    vecs.append((v, f"coord_w{ax + 1}"))
        elif self.mode == "grid2d":
            a, b = self.coordinate_axes[:2]
            for va in self.values:
                for vb in self.values:
                    v = list(self.anchor)
                    v[a] = va
                    v[b] = vb
                    vecs.append((v, f"grid2d_w{a + 1}xw{b + 1}"))
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        return vecs

    def generate(self) -> List[Candidate]:
        seen = set()
        out: List[Candidate] = []
        for raw, tag in self._raw_vectors():
            norm = normalize_mean(raw) if self.normalize_mean else list(map(float, raw))
            if self.monotonic and tag not in self.always_include and not is_monotonic(norm):
                continue
            k = _key(norm)
            if k in seen:
                continue
            seen.add(k)
            out.append(Candidate(id=f"cand_{len(out):03d}", raw=list(map(float, raw)),
                                  normalized=norm, tag=tag))
        # keep the always_include vectors even if the cap is hit
        forced = [c for c in out if c.tag in self.always_include]
        rest = [c for c in out if c.tag not in self.always_include]
        rest = rest[: max(0, self.max_candidates - len(forced))]
        merged = forced + rest
        for i, c in enumerate(merged):
            c.id = f"cand_{i:03d}"
        return merged


def from_config(grid_cfg: dict) -> SearchSpace:
    return SearchSpace(
        values=grid_cfg.get("values", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
        normalize_mean=grid_cfg.get("normalize_mean", True),
        monotonic=grid_cfg.get("monotonic", True),
        mode=grid_cfg.get("mode", "grid2d"),
        anchor=grid_cfg.get("anchor", [1.0] * N_BUCKETS),
        coordinate_axes=grid_cfg.get("coordinate_axes", [3, 4]),
        max_candidates=int(grid_cfg.get("max_candidates", 40)),
        always_include=grid_cfg.get("always_include", {}) or {},
    )
