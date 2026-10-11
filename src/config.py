"""
Configuration system for TLC_TALLIP.

Everything is YAML-driven (brief section: "Make all new experiments
configuration-driven rather than hard-coded").  A run config is composed from:

    configs/datasets/<dataset_version>.yaml   # data paths + frozen statistics
    configs/models/<model>.yaml               # base model + tokenizer/lang setup
    configs/experiments/<experiment>.yaml     # training budget + shared knobs
    + the X-ABCD case flags (A=BBS B=TLC C=RoPE D=ALiBi)
    + an explicit --seed
    + optional CLI/dict overrides

`RunConfig.resolve(...)` returns a fully-populated, JSON-serialisable object
that the trainer / evaluator consume.  Nothing else in the codebase should
hard-code a hyper-parameter.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

# ---------------------------------------------------------------------------
# X-ABCD case convention  (DO NOT CHANGE THE MEANING OF THE FOUR BITS)
#   position 0  A = BBS   (bucket-balanced sampling)
#   position 1  B = TLC   (token-length conditioning: control token + length loss)
#   position 2  C = RoPE
#   position 3  D = ALiBi
# ---------------------------------------------------------------------------
CASE_BITS = ("bbs", "tlc", "rope", "alibi")
FACTOR_LETTERS = {"bbs": "A", "tlc": "B", "rope": "C", "alibi": "D"}
FACTOR_DISPLAY = {"bbs": "BBS", "tlc": "TLC", "rope": "RoPE", "alibi": "ALiBi"}

ALL_CASE_IDS: List[str] = [f"X-{a}{b}{c}{d}" for a in "01" for b in "01" for c in "01" for d in "01"]
# canonical order matching the brief's listing (X-0000, X-0001, ... X-1111)


def parse_case_id(case_id: str) -> Dict[str, int]:
    """'X-1101' -> {'bbs':1,'tlc':1,'rope':0,'alibi':1}."""
    cid = case_id.strip().upper()
    if not cid.startswith("X-") or len(cid) != 6 or any(c not in "01" for c in cid[2:]):
        raise ValueError(f"Invalid case id {case_id!r}; expected X-ABCD with A..D in {{0,1}}")
    return {name: int(bit) for name, bit in zip(CASE_BITS, cid[2:])}


def case_label(case_id: str) -> str:
    flags = parse_case_id(case_id)
    on = [FACTOR_DISPLAY[name] for name in CASE_BITS if flags[name]]
    return " + ".join(on) if on else "baseline"


def format_case_id(bbs: int, tlc: int, rope: int, alibi: int) -> str:
    return f"X-{int(bool(bbs))}{int(bool(tlc))}{int(bool(rope))}{int(bool(alibi))}"


# ---------------------------------------------------------------------------
# The resolved run configuration
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    # ---- identity -------------------------------------------------------
    experiment: str = "factorial"           # experiment family (factorial / bbs_search / ...)
    case_id: str = "X-0000"
    seed: int = 42
    dataset_version: str = "v2"
    model_key: str = "nllb"                 # configs/models/<model_key>.yaml stem
    run_group: str = "factorial"            # sub-dir under experiments/<dataset>/<run_group>/

    # ---- X-ABCD factor switches --------------------------------------
    use_bucket_balanced_sampling: bool = False   # A = BBS
    use_length_control_token: bool = False       # B = TLC (token part)
    use_length_loss: bool = False                # B = TLC (loss part)  -- independently switchable
    use_rope: bool = False                       # C
    use_alibi: bool = False                      # D

    # ---- BBS ----------------------------------------------------------
    bucket_boundaries: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    bucket_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.2, 1.5, 2.0])
    bbs_deterministic: bool = True               # seed the WeightedRandomSampler generator

    # ---- TLC bucketing scheme ablation (reviewer Q3) ------------------
    bucket_mode: str = "source_length"           # source_length (default) |
                                                  # predicted_target_length | oracle_ratio_class
                                                  # -- see src/data/dataset.py::SaHiDataset
    target_length_ratio: float = 1.8017          # global mean(target_len/source_len); only used
                                                  # when bucket_mode == predicted_target_length
    predicted_target_boundaries: List[float] = field(default_factory=lambda: [9, 18, 27, 36])
    oracle_ratio_boundaries: List[float] = field(default_factory=lambda: [1.0, 1.4, 1.667, 2.0])

    # ---- training-data repetition (reuses the list-based augmentation idiom;
    #      the composed training list is replicated k times before the dataset /
    #      WeightedRandomSampler is built).  Used by the V1+a*V2 scale ablation. --
    train_repeat_factor: int = 1

    # ---- TLC length loss -------------------------------------------
    length_loss_lambda: float = 0.1             # section 21 default
    length_loss_epsilon: float = 1.0
    length_loss_type: str = "l1"                # l1 (default) | huber | squared | l1_minus1
                                                 # -- see src/models/length_loss.py for each formula
    bucket_embed_dim: Optional[int] = None      # None -> the 5 LEN_BUCKET_* tokens use the model's
                                                 # ordinary tied vocab embedding (default). Otherwise
                                                 # they get a DEDICATED nn.Embedding(5, bucket_embed_dim)
                                                 # + a learned linear projection to d_model, independent
                                                 # of the frozen pretrained d_model -- see
                                                 # src/models/bucket_embedding.py

    # ---- data -------------------------------------------------------
    data_dir: str = "data/v2"
    train_src: str = "train.sa"
    train_tgt: str = "train.hi"
    val_src: str = "val.sa"
    val_tgt: str = "val.hi"
    test_src: str = "test_7264_5.sa"
    test_tgt: str = "test_7264_5.hi"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    # when subsampling train, keep the full corpus's per-bucket proportions
    # (instead of a plain first-N slice).  Matters for the BBS search.
    stratified_train_subsample: bool = False
    src_lang: str = "san_Deva"                  # NLLB code (canonical internal repr)
    tgt_lang: str = "hin_Deva"
    dataset_fingerprint: Optional[str] = None   # filled from the dataset manifest

    # ---- model ----------------------------------------------------
    model_family: str = "nllb"                  # nllb | mbart | mt5 | indictrans2 | tf6 | tf12
    base_model: str = "facebook/nllb-200-distilled-600M"
    model_revision: Optional[str] = None
    # tokenizer source, when it differs from base_model (e.g. a from-scratch
    # tf6/tf12 model borrows the NLLB tokenizer's vocab; base_model itself
    # names no real checkpoint for those).  Defaults to base_model.
    tokenizer_base: Optional[str] = None
    scratch_layers: int = 6                     # encoder=decoder layer count for tf6/tf12
    scratch_d_model: int = 512
    scratch_ffn_dim: int = 2048
    scratch_attention_heads: int = 8
    # mBART-50 M2M language setup (section 5) — resolved from the model YAML
    mbart_src_lang: Optional[str] = None        # e.g. "hi_IN" (Sanskrit-as-Hindi, see YAML rationale)
    mbart_tgt_lang: Optional[str] = None        # e.g. "hi_IN"
    force_bos_target_lang: bool = True

    max_input_length: int = 512
    max_target_length: int = 128

    # ---- training budget ------------------------------------------
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    auto_find_batch_size: bool = False   # start at per_device_train_batch_size, halve on CUDA OOM until it fits
    gradient_checkpointing: bool = False
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    label_smoothing: float = 0.1
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0    # 0 = load in the main process; avoids torch.multiprocessing worker temp-dir (/tmp) failures under disk-quota pressure
    optimizer: str = "adamw_torch"

    # ---- eval / checkpoint selection (VALIDATION ONLY; section 28) ---
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_every_n_epochs: int = 1    # >1 -> eval + checkpoint every N epochs (uses a 'steps' schedule)
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"    # validation metric
    greater_is_better: bool = True
    comet_eval_every: int = 5
    train_eval_num_beams: int = 1
    final_eval_num_beams: int = 4
    logging_steps: int = 50
    early_stopping_patience: Optional[int] = None

    # ---- bookkeeping ------------------------------------------------
    output_dir: str = "experiments/v2/factorial/X-0000/seed_42"
    deterministic: bool = True
    notes: str = ""

    # =====================================================================
    @property
    def factors(self) -> Dict[str, int]:
        return parse_case_id(self.case_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True, allow_unicode=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    # =====================================================================
    @classmethod
    def resolve(
        cls,
        case_id: str,
        seed: int,
        dataset_version: str = "v2",
        model_key: str = "nllb",
        experiment: str = "factorial",
        run_group: Optional[str] = None,
        experiments_root: str = "experiments",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "RunConfig":
        """Compose dataset + model + experiment YAML + case flags + seed + overrides."""
        run_group = run_group or experiment
        cfg = cls(
            experiment=experiment,
            case_id=case_id.upper(),
            seed=int(seed),
            dataset_version=dataset_version,
            model_key=model_key,
            run_group=run_group,
        )

        # 1) experiment defaults (training budget, shared knobs)
        exp_yaml = os.path.join(CONFIGS_DIR, "experiments", f"{experiment}.yaml")
        if os.path.isfile(exp_yaml):
            _apply(cfg, _load_yaml(exp_yaml).get("defaults", {}))

        # 2) dataset manifest / config
        ds_yaml = os.path.join(CONFIGS_DIR, "datasets", f"{dataset_version}.yaml")
        if not os.path.isfile(ds_yaml):
            raise FileNotFoundError(
                f"Dataset config not found: {ds_yaml}. Run scripts/prepare_dataset.py first."
            )
        _apply(cfg, _dataset_section(_load_yaml(ds_yaml)))

        # 3) model config
        md_yaml = os.path.join(CONFIGS_DIR, "models", f"{model_key}.yaml")
        if not os.path.isfile(md_yaml):
            raise FileNotFoundError(f"Model config not found: {md_yaml}")
        _apply(cfg, _load_yaml(md_yaml))

        # 4) X-ABCD case flags  -> concrete switches
        f = parse_case_id(cfg.case_id)
        cfg.use_bucket_balanced_sampling = bool(f["bbs"])
        # TLC bit turns *both* sub-components on by default; the decomposition
        # ablation (section 19) overrides these two keys explicitly.
        cfg.use_length_control_token = bool(f["tlc"])
        cfg.use_length_loss = bool(f["tlc"])
        cfg.use_rope = bool(f["rope"])
        cfg.use_alibi = bool(f["alibi"])

        # 5) architecture-driven budget adjustments (mirrors TLC_multi)
        if cfg.use_rope and cfg.max_input_length < 1024:
            cfg.max_input_length = 1024
        if cfg.use_alibi:
            # keep effective batch size constant vs non-ALiBi runs
            cfg.per_device_train_batch_size = max(1, cfg.per_device_train_batch_size // 2)
            cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps * 2
            cfg.gradient_checkpointing = True
            if model_key == "mbart50_many_to_many" and cfg.fp16:
                # mBART's eager-attention ALiBi path overflows fp16 (grad_norm
                # NaN from step 1, optimizer.step() a no-op every step, no real
                # training happens -> near-zero BLEU that the auto failure
                # check can miss if it's not exactly 0.0).  Confirmed on
                # X-0001/seed_42 (2026-09-23): eval_bleu 1.57 during training,
                # test BLEU 7.1 -- essentially untrained.  NLLB's ALiBi configs
                # train fine under fp16, so this is mBART-specific; bf16 fixes
                # it the same way it fixed mT5 (see baseline_mt5.yaml).
                cfg.fp16 = False
                cfg.bf16 = True

        # 6) explicit overrides win over everything
        if overrides:
            _apply(cfg, overrides)

        # 7) deterministic output directory
        cfg.output_dir = os.path.join(
            experiments_root, dataset_version, run_group, cfg.case_id, f"seed_{cfg.seed}"
        )
        return cfg


# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_section(d: dict) -> dict:
    """Flatten a dataset manifest YAML into RunConfig keys."""
    out = {}
    paths = d.get("paths", {})
    stats = d.get("statistics", {})
    for k in ("data_dir", "train_src", "train_tgt", "val_src", "val_tgt", "test_src", "test_tgt"):
        if k in paths:
            out[k] = paths[k]
    if "src_lang" in d:
        out["src_lang"] = d["src_lang"]
    if "tgt_lang" in d:
        out["tgt_lang"] = d["tgt_lang"]
    if "fingerprint" in d:
        out["dataset_fingerprint"] = d["fingerprint"]
    elif "fingerprint" in stats:
        out["dataset_fingerprint"] = stats["fingerprint"]
    return out


def _apply(cfg: RunConfig, updates: dict) -> None:
    known = {f.name for f in fields(RunConfig)}
    for k, v in (updates or {}).items():
        if k in known:
            setattr(cfg, k, v)
        # silently ignore unknown keys so YAML can carry documentation blocks


def load_experiment_yaml(name: str) -> dict:
    """Full experiment YAML (used by the runners for their own settings)."""
    path = os.path.join(CONFIGS_DIR, "experiments", f"{name}.yaml")
    return _load_yaml(path)


def load_dataset_config(version: str) -> dict:
    """Full dataset manifest YAML for a version ('v1' / 'v2')."""
    path = os.path.join(CONFIGS_DIR, "datasets", f"{version}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset config not found: {path} (run scripts/prepare_dataset.py --version {version})")
    return _load_yaml(path)


def dataset_test_variants(version: str) -> Dict[str, Dict[str, str]]:
    """
    {'sampled': {'src':..,'tgt':..}, 'unsampled': {'src':'test.sa','tgt':'test.hi'}}
    for a dataset version, from its manifest's `test_variants` block (falls back
    to the single `paths.test_src/test_tgt` as 'sampled').
    """
    d = load_dataset_config(version)
    tv = d.get("test_variants")
    if tv:
        return {k: {"src": v["src"], "tgt": v["tgt"]} for k, v in tv.items()}
    p = d.get("paths", {})
    return {"sampled": {"src": p.get("test_src"), "tgt": p.get("test_tgt")}}


def dataset_data_dir(version: str) -> str:
    return load_dataset_config(version)["paths"]["data_dir"]
