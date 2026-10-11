#!/usr/bin/env python
"""
Formal main-effects / interaction decomposition of the 2^4 (A=BBS, B=TLC,
C=RoPE, D=ALiBi) x 3-seed factorial, to put the paper's "super-additive"
claim on statistical footing (reviewer Q6) -- currently the paper shows that
the all-on config (X-1111) beats its individual components, which is NOT
by itself evidence of super-additivity: it's also what plain ADDITIVE
effects would predict if all four factors individually help. A formal test
needs the actual INTERACTION terms of a factorial model.

Method
------
48 observations per (architecture, metric): 16 configs x 3 seeds (seeds are
genuine independent replicates of the exact same config -- this is what
supplies the residual degrees of freedom a classic unreplicated 2^4 design
would lack). Factors coded +-1 (standard factorial coding -> an orthogonal
design matrix, so every effect estimate is unconfounded).

Two models are fit by OLS:
  reduced  y ~ 1 + A + B + C + D                         (purely additive)
  full     y ~ 1 + A + B + C + D + all 11 interactions    (saturated, 16 df)

For every one of the 15 non-intercept effects in the full model: coefficient,
SE, t, p (two-sided), and p_holm (Holm-Bonferroni across all 15, within each
(arch, metric) -- the natural family for "which effects are real").

Super-additivity test (the actual answer to Q6): an F-test comparing the
full vs reduced model (nested, classic ANOVA) -- does including ALL 11
interaction terms explain significantly more variance than the 4 main
effects alone? Also reported concretely at the X-1111 corner: observed
mean(X-1111) vs the reduced (additive-only) model's predicted value there
vs the full model's predicted value there -- the gap between additive and
full IS the formal "interaction surplus" the abstract's claim rests on.

Data: experiments/v2/{factorial,mbart50}/X-*/seed_{42,43,44}/evaluation/v2/
unsampled/metrics.json (the main 48-run factorials, already completed).
"""
import _paths  # noqa: F401

import itertools
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import ALL_CASE_IDS, parse_case_id
from src.utils.io import ensure_dir

ARCHS = [
    {"name": "nllb", "run_group": "factorial"},
    {"name": "mbart50", "run_group": "mbart50"},
]
SEEDS = [42, 43, 44]
METRICS = ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"]
FACTOR_NAMES = ["A", "B", "C", "D"]  # BBS, TLC, RoPE, ALiBi
OUT_ROOT = "results/factorial_interaction_analysis"


def load_long_table(run_group):
    rows = []
    for cid in ALL_CASE_IDS:
        f = parse_case_id(cid)
        coded = {name: (1 if f[key] else -1)
                for name, key in zip(FACTOR_NAMES, ["bbs", "tlc", "rope", "alibi"])}
        for seed in SEEDS:
            p = f"experiments/v2/{run_group}/{cid}/seed_{seed}/evaluation/v2/unsampled/metrics.json"
            if not os.path.isfile(p):
                continue
            m = json.load(open(p))["metrics"]
            row = {"config": cid, "seed": seed, **coded}
            for metric in METRICS:
                row[metric] = m.get(metric)
            rows.append(row)
    return pd.DataFrame(rows)


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running_max)
    return adj


def interaction_terms():
    """All 11 interaction column names/orders for the full model, order 2..4."""
    terms = []
    for order in (2, 3, 4):
        for combo in itertools.combinations(FACTOR_NAMES, order):
            terms.append(combo)
    return terms


def add_interaction_columns(df):
    df = df.copy()
    for combo in interaction_terms():
        name = "".join(combo)
        prod = df[combo[0]].copy()
        for c in combo[1:]:
            prod = prod * df[c]
        df[name] = prod
    return df


def fit(df, metric, cols):
    y = df[metric].astype(float)
    X = sm.add_constant(df[cols].astype(float))
    model = sm.OLS(y, X).fit()
    return model


def analyze_one(arch_name, df, metric, out_dir):
    df = df.dropna(subset=[metric])
    df = add_interaction_columns(df)
    inter_cols = ["".join(c) for c in interaction_terms()]
    full_cols = FACTOR_NAMES + inter_cols
    reduced_cols = FACTOR_NAMES

    full_model = fit(df, metric, full_cols)
    reduced_model = fit(df, metric, reduced_cols)

    # per-effect table from the FULL model
    effect_rows = []
    pvals = full_model.pvalues.drop("const").values
    p_holm = holm_bonferroni(pvals)
    for i, name in enumerate(full_model.params.index.drop("const")):
        effect_rows.append({
            "arch": arch_name, "metric": metric, "effect": name,
            "coefficient": round(float(full_model.params[name]), 4),
            "se": round(float(full_model.bse[name]), 4),
            "t": round(float(full_model.tvalues[name]), 3),
            "p": round(float(full_model.pvalues[name]), 5),
            "p_holm": round(float(p_holm[i]), 5),
        })

    # nested F-test: full vs reduced (the formal super-additivity test)
    f_test = full_model.compare_f_test(reduced_model)
    f_stat, f_pvalue, df_diff = f_test

    # concrete corner: X-1111 (all factors +1)
    observed_1111 = df.loc[df["config"] == "X-1111", metric].mean()
    corner_full = {**{c: 1.0 for c in FACTOR_NAMES}, **{c: 1.0 for c in inter_cols}}
    corner_reduced = {c: 1.0 for c in FACTOR_NAMES}
    pred_full = float(full_model.predict(sm.add_constant(
        pd.DataFrame([corner_full])[full_cols], has_constant="add"))[0])
    pred_additive = float(reduced_model.predict(sm.add_constant(
        pd.DataFrame([corner_reduced])[reduced_cols], has_constant="add"))[0])
    interaction_surplus = pred_full - pred_additive

    summary = {
        "arch": arch_name, "metric": metric,
        "n_obs": int(len(df)), "residual_df_full": int(full_model.df_resid),
        "f_stat_full_vs_additive": round(float(f_stat), 4),
        "f_pvalue_full_vs_additive": float(f_pvalue),
        "observed_mean_X1111": round(float(observed_1111), 4),
        "additive_model_predicted_X1111": round(pred_additive, 4),
        "full_model_predicted_X1111": round(pred_full, 4),
        "interaction_surplus_at_X1111": round(interaction_surplus, 4),
        "r2_full": round(float(full_model.rsquared), 4),
        "r2_reduced": round(float(reduced_model.rsquared), 4),
    }
    return effect_rows, summary


def main():
    ensure_dir(OUT_ROOT)
    all_effects, all_summaries = [], []

    for arch in ARCHS:
        df = load_long_table(arch["run_group"])
        n_cells = df.groupby("config").size()
        print(f"\n=== {arch['name']}: {len(df)} rows, "
              f"{(n_cells == 3).sum()}/16 configs with all 3 seeds ===")
        for metric in METRICS:
            effect_rows, summary = analyze_one(arch["name"], df, metric, OUT_ROOT)
            all_effects.extend(effect_rows)
            all_summaries.append(summary)

            sig = [r["effect"] for r in effect_rows if r["p_holm"] < 0.05]
            print(f"  {metric:<10s} F(full vs additive)={summary['f_stat_full_vs_additive']:.3f}  "
                  f"p={summary['f_pvalue_full_vs_additive']:.5f}  "
                  f"interaction_surplus@X1111={summary['interaction_surplus_at_X1111']:+.3f}  "
                  f"Holm-sig effects={sig if sig else 'none'}")

    import csv
    with open(os.path.join(OUT_ROOT, "effects.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_effects[0].keys()))
        w.writeheader()
        w.writerows(all_effects)
    with open(os.path.join(OUT_ROOT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        w.writeheader()
        w.writerows(all_summaries)
    print(f"\nwrote {OUT_ROOT}/effects.csv ({len(all_effects)} rows) "
          f"and {OUT_ROOT}/summary.csv ({len(all_summaries)} rows)")

    print(f"\n{'=' * 110}\n  SUPER-ADDITIVITY TEST  (F-test: full model incl. all 11 interactions vs additive-only "
          f"A+B+C+D)\n{'=' * 110}")
    hdr = (f"  {'arch':8s} {'metric':10s} {'F':>8s} {'p(F)':>9s} {'obs X-1111':>11s} "
           f"{'additive pred':>14s} {'full pred':>10s} {'surplus':>9s} {'r2_full':>8s} {'r2_add':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in all_summaries:
        sig_mark = "*" if s["f_pvalue_full_vs_additive"] < 0.05 else " "
        print(f"  {s['arch']:8s} {s['metric']:10s} {s['f_stat_full_vs_additive']:8.3f} "
              f"{s['f_pvalue_full_vs_additive']:9.5f}{sig_mark} {s['observed_mean_X1111']:11.3f} "
              f"{s['additive_model_predicted_X1111']:14.3f} {s['full_model_predicted_X1111']:10.3f} "
              f"{s['interaction_surplus_at_X1111']:+9.3f} {s['r2_full']:8.3f} {s['r2_reduced']:7.3f}")
    print("\n  (* = p<0.05: the interaction terms collectively explain significantly more variance\n"
          "     than the 4 main effects alone -- this IS the formal super-additivity test.)\n")


if __name__ == "__main__":
    main()
