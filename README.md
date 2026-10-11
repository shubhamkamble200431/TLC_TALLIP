# Learning the Decoder When to Stop

Length-aware Sanskrit machine translation: a **Target-Length Conditioning
(TLC)** mechanism that tells the decoder, via a length-bucket signal, roughly
how long the output should be, paired with a differentiable length-aware loss
term. Evaluated on Sanskrit→Hindi, Sanskrit→Gujarati, and Sanskrit→Marathi,
across NLLB-200-distilled-600M and mBART-50-many-to-many, with ablations
against RoPE and ALiBi positional-bias variants and a battery of classical
seq2seq baselines (mT5, Transformer-6/12/18, Mamba).

This repository contains **code and datasets only** — no model checkpoints,
logs, SLURM job scripts, or precomputed result tables/figures. Every result
is reproducible by re-running the scripts below against the included data.

## Factorial convention

Experiment configs are named `X-ABCD`, a 4-bit switch encoding:

```
A = BBS (bucket-balanced sampling)
B = TLC (target-length conditioning + length-aware loss)
C = RoPE (rotary positional embeddings)
D = ALiBi (attention with linear biases)
```

e.g. `X-0100` = TLC only, `X-0010` = RoPE only, `X-1111` = all four combined.
Source-length buckets throughout: `B1`=1–5 tokens, `B2`=6–10, `B3`=11–15,
`B4`=16–20, `B5`=21+ (whitespace-token count on the Sanskrit side).

A completed run (not included here) would write to
`experiments/<dataset>/<group>/<X-ABCD>/seed_<n>/` (config, manifest, status,
checkpoints, per-split evaluation + metrics) and `results/{csv,json,tables,
figures}/`; re-running the scripts below regenerates both from scratch.

## Setup

```bash
conda create -n nmt_tlc python=3.10
conda activate nmt_tlc
pip install -r requirements.txt
```

Verified working set: Python 3.10, CUDA 12.4, torch 2.6, transformers 4.57.

## Quick start

```bash
# 0. build dataset manifests + fingerprints from data/v2 and data/v1
python scripts/prepare_dataset.py --version v2
python scripts/prepare_dataset.py --version v1
python scripts/compare_datasets.py          # structure + bucket + overlap/leakage report

# 1. run the full 16-config x 3-seed factorial for one architecture
python scripts/run_factorial.py --model nllb      --config configs/experiments/factorial.yaml
python scripts/run_mbart50.py   --config configs/experiments/mbart50.yaml

# 2. evaluate on the out-of-domain news test set
python scripts/evaluate_news_test.py --models nllb mbart50

# 3. aggregate + statistical testing
python scripts/factorial_interaction_analysis.py
python scripts/bootstrap_significance_v2.py

# 4. tables and figures
python scripts/generate_tables.py
python scripts/generate_figures.py
```

---

## Structure

```
data/
├── v1/                       older/smaller Sanskrit-Hindi corpus (train/val/test .sa+.hi)
├── v2/                       main Sanskrit-Hindi corpus (train/val/test .sa+.hi)
├── news_test/                out-of-domain eval set: shared .sa source, .hi/.gu/.mr references,
│                              test_manifest.tsv (per-sentence bucket B1-B5)
└── indic_data/
    ├── san-guj/               Sanskrit-Gujarati train/val/test + split_manifest.json
    └── san-mar/               Sanskrit-Marathi train/val/test + split_manifest.json

configs/
├── datasets/
│   ├── v1.yaml                frozen fingerprint + bucket boundaries for data/v1
│   └── v2.yaml                frozen fingerprint + bucket boundaries for data/v2
├── models/
│   ├── nllb.yaml               NLLB-200-distilled-600M (primary backbone)
│   ├── mbart50_many_to_many.yaml  mBART-50-M2M (primary backbone)
│   ├── mt5_base.yaml            mT5 baseline
│   ├── tf6.yaml / tf12.yaml / tf18.yaml   scratch-trained Transformers (6/12/18 layers,
│   │                            NOT pretrained — base_model is a marker string, not an HF repo)
│   └── indictrans2_indic_indic_1b.yaml    GATED model, needs HF access approval + token
└── experiments/
    ├── factorial.yaml / factorial_16.yaml      the main 16x3 NLLB factorial (identical content,
    │                                            two filenames some scripts expect)
    ├── mbart50.yaml                             same 16x3 factorial, mBART-50
    ├── baseline_{nllb,mbart,mt5,tf6,tf12,tf18,indictrans2}.yaml   X-0000-only, generic-recipe
    │                                            baseline per architecture (one file each)
    ├── bbs_grid.yaml / bbs_search.yaml / bbs_search_mbart.yaml    BBS weight-vector search
    │                                            (validation-only; _mbart = mBART-50 variant)
    ├── bucket_repeat.yaml                        bucket-cap x repetition inference ablation
    ├── domain_length.yaml                        domain x length-bucket eval
    ├── tlc_ablation.yaml / tlc_lambda.yaml        TLC loss-lambda sensitivity (both drive
    │                                            scripts/run_tlc_ablation.py)
    └── v1_v2_scale.yaml                          V1+alpha*V2 data-scale ablation

src/
├── config.py                  YAML -> RunConfig: the single source of truth for a run
├── data/
│   ├── buckets.py              B1-B5 source-length bucket definitions
│   ├── dataset.py               SaHiDataset + bucket-balanced sampler
│   ├── manifest.py              dataset fingerprinting / drift detection
│   └── v1v2.py                  V1 + alpha*V2 composite training-set construction
├── models/
│   ├── builder.py                unified model+tokenizer construction (all architectures)
│   ├── bucket_embedding.py       optional dedicated embedding for the 5 length-bucket tokens
│   ├── lang_codes.py             Sanskrit/Hindi/Gujarati/Marathi language-code resolution
│   ├── length_loss.py            differentiable TLC length loss (factor B)
│   └── positional.py             RoPE (factor C) + ALiBi (factor D)
├── training/
│   └── trainer.py                Seq2SeqTrainer wrapper: BBS sampler + TLC loss + history dump
├── evaluation/
│   ├── generate.py               unified translate() for NLLB/mBART-50
│   ├── metrics.py                BLEU/chrF2/TER/METEOR/BERTScore/COMET + length metrics, per-bucket
│   ├── hf_metrics.py             validation-only compute_metrics callback for the HF Trainer
│   └── evaluate.py               held-out test evaluation orchestration
├── experiments/
│   ├── factorial.py              expands the 16x3 factorial plan into concrete runs
│   ├── bbs_grid.py               BBS search-space generation
│   ├── train_run.py              core per-run training orchestrator
│   ├── status.py                 run lifecycle tracking (started/running/completed/failed)
│   ├── manifest.py               per-run manifest JSON
│   └── failure.py                post-run automatic failure detection
├── statistics/
│   ├── aggregate.py               3-seed mean/std/95% CI aggregation
│   ├── bootstrap.py               paired bootstrap + Holm-Bonferroni
│   └── factorial.py                2^4 factorial OLS fit (effects + interactions)
├── reporting/
│   └── tables.py                  LaTeX table generation from CSV/JSON
├── visualization/
│   ├── heatmaps.py / plots.py      publication figures from saved CSVs
│   └── style.py                    shared matplotlib style
└── utils/
    ├── seeding.py                  centralized seed control (reproducibility)
    ├── env_info.py / gpu.py / io.py   provenance capture, GPU discovery, fs/JSON helpers

scripts/
├── _paths.py                          sys.path shim imported by every CLI script (not a CLI itself)
├── prepare_dataset.py / compare_datasets.py       dataset manifest + V1-vs-V2 comparison
├── run_factorial.py / run_mbart50.py              the two main 16x3 factorial runners
│                                                   (run_mbart50 is a thin alias of run_factorial)
├── run_bbs_search.py / bbs_compare.py              BBS search, then manual-vs-grid comparison
├── run_tlc_ablation.py                             TLC lambda sensitivity ablation
├── run_bucket_repeat_ablation.py                   bucket-cap x repetition inference ablation
├── run_domain_length.py                            domain x length-bucket eval
├── run_v1_v2_scale_ablation.py                      V1+alpha*V2 data-scale ablation
├── train_mamba_baseline.py                          Mamba (decoder-only prefix-LM) baseline
├── _timing_test_mamba.py                            ad-hoc one-off benchmark script, not a real CLI
├── evaluate_news_test.py                            MAIN out-of-domain eval: all NLLB/mBART-50 runs
├── evaluate_news_test_crosslingual.py               same checkpoints, zero-shot into gu/mr
├── evaluate_news_test_indic.py                      checkpoints actually trained on san-guj/san-mar
├── evaluate_news_test_mamba.py                      separate eval path for the Mamba baselines
├── evaluate_news_test_tlc_ablation.py               eval for the TLC-variant ablation runs
├── evaluate_all.py                                  in-domain (v1/v2) eval + aggregation
├── best_checkpoint.py                               prints a checkpoint path (for shell capture)
├── bootstrap_significance.py / bootstrap_significance_v2.py   v2 supersedes v1 (full corpus, 6 metrics)
├── factorial_analysis.py / factorial_interaction_analysis.py  effects-only vs full interaction test
├── length_behavior_table.py                          assembles existing per-run metrics, no new compute
├── build_human_eval_template.py                      blank human-eval collection spreadsheet
├── generate_tables.py / generate_figures.py           LaTeX tables / publication figures
├── generate_bucket_repeat_heatmaps.py / generate_v1_v2_heatmaps.py / plot_dataset_histograms.py
├── generate_jobs.py                                   emits (never launches) job scripts
├── orchestrate_mbart_chain.py                         stateful cron-driven pipeline sequencer
├── track_runs.py                                      live run-status tracker spreadsheet
├── validate_experiments.py                            repo-wide invariant checks
└── prune_checkpoints.py                                disk-space cleanup (keeps checkpoint-best/)

tests/
├── conftest.py                       pytest sys.path bootstrap
├── test_config_and_cases.py          X-ABCD parsing + test-set isolation
├── test_seeding_and_buckets.py       seed reproducibility + bucket assignment
├── test_bbs.py                       BBS normalization/monotonicity/sampler
├── test_length_loss.py               differentiable TLC loss gradient checks
├── test_mbart_and_manifest.py        mBART language-code mapping + manifest invariants
├── test_bucket_repeat.py             bucket-cap x repetition ablation correctness
├── test_v1v2.py                      V1+alpha*V2 composition logic
└── test_stats_and_viz.py             aggregation, bootstrap, factorial design, heatmaps
```



### Metrics

| Metric | Description | Target |
|----------|-------------|----------|
| BLEU | corpus_bleu (FLORES-200 tokenizer) | Higher is better |
| chrF2 | Character n-gram F-score with word order information | Higher is better |
| TER | Translation Edit Rate; measures the number of edits required to transform a hypothesis into the reference translation | Lower is better |
| METEOR | Alignment-based metric incorporating exact, stem, synonym, and paraphrase matches | Higher is better |
| COMET | Neural MT evaluation metric (wmt22-comet-da) | Higher is better |
| BERTScore-F1 (BSF) | Semantic similarity measured using contextual embeddings and reported as the BERTScore F1 value | Higher is better |
