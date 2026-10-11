"""
Analysis of the replicated 2^4 factorial design (brief section 18).

Factors (contrast-coded -1 / +1):
    A = BBS    B = TLC    C = RoPE    D = ALiBi

Saturated model:
    Y = b0 + bA A + ... + bD D
          + bAB AB + ... + bCD CD
          + bABC ABC + ... + bBCD BCD
          + bABCD ABCD

Fitted by ordinary least squares over the replicated (3-seed) results.  For
each term we report the regression coefficient, the conventional factorial
"effect" (= 2 x coefficient for +/-1 coding), a standard error, a 95%
confidence interval and a p-value.

Wording guard (brief section 18 / 36): this module never labels an interaction
"super-additive".  It reports "positive interaction" / "negative interaction"
and whether it is statistically significant; interpretation is left to the
paper.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.config import parse_case_id

try:
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None

FACTORS = ["A", "B", "C", "D"]
FACTOR_TO_BIT = {"A": "bbs", "B": "tlc", "C": "rope", "D": "alibi"}


def term_names() -> List[str]:
    names = ["Intercept"]
    for k in range(1, 5):
        for combo in itertools.combinations(FACTORS, k):
            names.append("".join(combo))
    return names  # 1 + 4 + 6 + 4 + 1 = 16


def _row_for_case(case_id: str) -> Dict[str, int]:
    f = parse_case_id(case_id)
    return {ltr: (1 if f[FACTOR_TO_BIT[ltr]] else -1) for ltr in FACTORS}


def design_row(case_id: str) -> np.ndarray:
    coded = _row_for_case(case_id)
    row = [1.0]
    for k in range(1, 5):
        for combo in itertools.combinations(FACTORS, k):
            val = 1.0
            for ltr in combo:
                val *= coded[ltr]
            row.append(val)
    return np.array(row, dtype=float)


def build_design(case_ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    X = np.vstack([design_row(c) for c in case_ids])
    return X, term_names()


@dataclass
class Effect:
    term: str
    coefficient: float
    effect: float          # 2 x coefficient
    std_error: float
    ci_low: float
    ci_high: float
    t_value: float
    p_value: float
    df_resid: int
    direction: str         # "positive" | "negative" | "n/a" (intercept)
    significant: bool

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def fit(metric_by_config: Dict[str, List[float]], alpha: float = 0.05) -> List[Effect]:
    """
    metric_by_config : {case_id: [seed_1_value, seed_2_value, ...]}  (>=1 per config)
    """
    case_ids: List[str] = []
    y: List[float] = []
    for cid, vals in metric_by_config.items():
        for v in vals:
            if v is None or not np.isfinite(v):
                continue
            case_ids.append(cid)
            y.append(float(v))
    if len(set(case_ids)) < 16:
        missing = sorted(set(f"X-{a}{b}{c}{d}" for a in "01" for b in "01" for c in "01" for d in "01")
                         - set(case_ids))
        raise ValueError(f"factorial fit needs all 16 configs; missing: {missing}")

    X, names = build_design(case_ids)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - p
    sigma2 = float(resid @ resid) / df if df > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))

    if _sps is not None and df > 0:
        tcrit = float(_sps.t.ppf(1 - alpha / 2, df))
        pvals = [2 * (1 - _sps.t.cdf(abs(b / s), df)) if s > 0 else float("nan")
                 for b, s in zip(beta, se)]
    else:  # normal approx
        tcrit = 1.96
        from math import erf, sqrt

        pvals = [2 * (1 - 0.5 * (1 + erf(abs(b / s) / sqrt(2)))) if s > 0 else float("nan")
                 for b, s in zip(beta, se)]

    effects = []
    for i, name in enumerate(names):
        b = float(beta[i])
        s = float(se[i])
        pv = float(pvals[i])
        if name == "Intercept":
            direction = "n/a"
        else:
            direction = "positive" if b > 0 else "negative" if b < 0 else "zero"
        effects.append(
            Effect(
                term=name,
                coefficient=b,
                effect=2.0 * b,
                std_error=s,
                ci_low=b - tcrit * s,
                ci_high=b + tcrit * s,
                t_value=b / s if s > 0 else float("nan"),
                p_value=pv,
                df_resid=df,
                direction=direction,
                significant=bool(np.isfinite(pv) and pv < alpha and name != "Intercept"),
            )
        )
    return effects
