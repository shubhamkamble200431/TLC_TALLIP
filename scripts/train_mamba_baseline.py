#!/usr/bin/env python
"""
Mamba baseline (decoder-only prefix-LM), X-0000 only, 3 size tiers.

transformers has no encoder-decoder Mamba -- only MambaForCausalLM (plain
causal LM). Sanskrit->Hindi translation is therefore framed the standard
"decoder-only seq2seq" way: concatenate SOURCE <sep> TARGET into one
sequence, train with a causal mask over the whole thing, and mask the loss
(-100) on every position up to and including <sep> so only the target
tokens contribute to the LM loss. At inference, the model is fed
SOURCE <sep> and left to generate a continuation.

<sep> = the tokenizer's own eos_token_id (a common, simple prefix-LM
convention -- no new special token is introduced).

From scratch (random MambaConfig init, no pretrained backbone), reusing the
NLLB tokenizer purely for subword segmentation -- same philosophy as
tf6/tf12/tf18 (src/models/builder.py::_build_scratch_transformer): a weak,
non-pretrained baseline for comparison, not a competitive model.

Size tiers (SIZES below): small/base/large, scaling hidden_size and
num_hidden_layers only; state_size/conv_kernel/expand use MambaConfig's own
defaults throughout.

Checkpoint selection: eval LOSS (not generation-based BLEU as the other
baselines use) -- generation-based eval every epoch would mean running
`.generate()` over the full val set every epoch, matching the other
baselines' cost, but val loss is sufficient for a from-scratch weak
baseline and keeps this script self-contained. This is a deliberate
simplification, not an oversight -- the final, reported numbers always come
from the SAME full generation-based `compute_all` pipeline every other
model in this project uses.

Output layout matches every other run in this project (status.json,
config.yaml, checkpoint-best/, evaluation/v2/unsampled/{source,
references,predictions,metrics.json}):
    experiments/v2/baselines/mamba_<size>/X-0000/seed_<seed>/
"""
import _paths  # noqa: F401

import argparse
import json
import os
import time

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, MambaConfig, MambaForCausalLM,
    Trainer, TrainingArguments,
)

from src.data.dataset import load_parallel_corpus
from src.evaluation.metrics import compute_all
from src.utils.io import ensure_dir, write_json, write_lines

TOKENIZER_BASE = "facebook/nllb-200-distilled-600M"
SIZES = {
    "small": {"hidden_size": 512, "num_hidden_layers": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 16},
}
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
EPOCHS = 7            # matches every other baseline's generic HF recipe
LR = 5.0e-5
NUM_BEAMS = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset-version", default="v2")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-completed", action="store_true")
    return ap.parse_args()


def dataset_paths(dataset_version):
    with open(os.path.join("configs", "datasets", f"{dataset_version}.yaml")) as f:
        manifest = yaml.safe_load(f)
    return manifest["paths"]["data_dir"], manifest["paths"]["test_src"], manifest["paths"]["test_tgt"]


class PrefixLMDataset(Dataset):
    def __init__(self, srcs, tgts, tokenizer, sep_id):
        self.srcs, self.tgts = srcs, tgts
        self.tok = tokenizer
        self.sep_id = sep_id

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        src_ids = self.tok(self.srcs[idx], truncation=True, max_length=MAX_INPUT_LENGTH,
                            add_special_tokens=True)["input_ids"]
        tgt_ids = self.tok(self.tgts[idx], truncation=True, max_length=MAX_TARGET_LENGTH,
                            add_special_tokens=True)["input_ids"]
        input_ids = src_ids + [self.sep_id] + tgt_ids
        labels = [-100] * (len(src_ids) + 1) + list(tgt_ids)
        return {"input_ids": input_ids, "labels": labels}


class PrefixLMCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            n = len(ex["input_ids"])
            pad = max_len - n
            input_ids.append(ex["input_ids"] + [self.pad_id] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_model(tokenizer, size):
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    mconf = MambaConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=bos,
        eos_token_id=tokenizer.eos_token_id,
        use_mambapy=True,  # pure-PyTorch parallel scan (mambapy pkg) -- no nvcc/CUDA-kernel
                           # build available on this cluster, and transformers' own fallback
                           # (plain sequential Python loop over every timestep) was ~20h/run;
                           # mambapy's chunked parallel scan is far faster at the same memory.
        **SIZES[size],
    )
    return MambaForCausalLM(mconf)


@torch.no_grad()
def generate_predictions(model, tokenizer, srcs, sep_id, device, batch_size=16):
    model.eval()
    preds = []
    for i in range(0, len(srcs), batch_size):
        chunk = srcs[i:i + batch_size]
        enc = [tokenizer(s, truncation=True, max_length=MAX_INPUT_LENGTH,
                          add_special_tokens=True)["input_ids"] + [sep_id] for s in chunk]
        max_len = max(len(e) for e in enc)
        # LEFT-pad the prompt so every sequence's <sep> (and generation start) lines
        # up at the same position -- required for batched causal-LM generation.
        input_ids = torch.tensor(
            [[tokenizer.pad_token_id] * (max_len - len(e)) + e for e in enc], dtype=torch.long
        ).to(device)
        attn = torch.tensor(
            [[0] * (max_len - len(e)) + [1] * len(e) for e in enc], dtype=torch.long
        ).to(device)
        out = model.generate(input_ids=input_ids, attention_mask=attn,
                              max_new_tokens=MAX_TARGET_LENGTH, num_beams=NUM_BEAMS,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for row in out[:, max_len:]:
            text = tokenizer.decode(row, skip_special_tokens=True)
            preds.append(text)
        print(f"    generated {min(i + batch_size, len(srcs))}/{len(srcs)}")
    return preds


def main():
    args = parse_args()
    run_dir = os.path.join(args.experiments_root, args.dataset_version,
                            "baselines", f"mamba_{args.size}", "X-0000", f"seed_{args.seed}")
    status_path = os.path.join(run_dir, "status.json")
    if args.skip_completed and os.path.isfile(status_path):
        if json.load(open(status_path)).get("status") == "completed":
            print(f"[skip] {run_dir} already completed")
            return

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir, test_src_file, test_tgt_file = dataset_paths(args.dataset_version)
    train_src, train_tgt = load_parallel_corpus(data_dir, "train.sa", "train.hi")
    val_src, val_tgt = load_parallel_corpus(data_dir, "val.sa", "val.hi")
    test_src, test_tgt = load_parallel_corpus(data_dir, "test.sa", "test.hi")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_BASE, use_fast=True,
                                              src_lang="san_Deva", tgt_lang="hin_Deva")
    sep_id = tokenizer.eos_token_id

    resume_ckpt = None
    ckpt_root = os.path.join(run_dir, "checkpoint")
    if args.resume and os.path.isdir(ckpt_root):
        subdirs = [d for d in os.listdir(ckpt_root) if d.startswith("checkpoint-")]
        if subdirs:
            resume_ckpt = os.path.join(ckpt_root, sorted(subdirs, key=lambda d: int(d.split("-")[1]))[-1])

    model = build_model(tokenizer, args.size)
    print(f"[mamba_{args.size}] seed={args.seed}  params={sum(p.numel() for p in model.parameters()):,}")

    train_ds = PrefixLMDataset(train_src, train_tgt, tokenizer, sep_id)
    val_ds = PrefixLMDataset(val_src, val_tgt, tokenizer, sep_id)
    collator = PrefixLMCollator(tokenizer.pad_token_id)

    targs = TrainingArguments(
        output_dir=ckpt_root,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=True,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        weight_decay=0.0,
        warmup_steps=0,
        lr_scheduler_type="linear",
        fp16=False,
        bf16=torch.cuda.is_available(),  # Mamba's SSM recurrence overflows fp16's narrow range
                                         # (confirmed: fp16 collapsed every run so far -- loss
                                         # explodes, grad_norm -> nan, GradScaler then silently
                                         # skips every step, loss pins at exactly 0.0). Same fix
                                         # already applied to mT5 and mBART+ALiBi this session.
        dataloader_num_workers=0,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=collator)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"[mamba_{args.size}] training done in {time.time() - t0:.0f}s")

    best_dir = os.path.join(run_dir, "checkpoint-best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    model.to(device)
    preds = generate_predictions(model, tokenizer, test_src, sep_id, device, batch_size=args.batch_size)
    scores = compute_all(preds, test_tgt, test_src, metrics=["bleu", "chrf2", "ter", "meteor",
                                                              "bertscore", "comet"],
                         run_comet=True, max_gen_len=MAX_TARGET_LENGTH, bertscore_lang="hi")

    eval_dir = os.path.join(run_dir, "evaluation", args.dataset_version, "unsampled")
    ensure_dir(eval_dir)
    write_lines(os.path.join(eval_dir, "source"), test_src)
    write_lines(os.path.join(eval_dir, "references"), test_tgt)
    write_lines(os.path.join(eval_dir, "predictions"), preds)
    write_json(os.path.join(eval_dir, "metrics.json"), {
        "model": f"mamba_{args.size}", "case_id": "X-0000", "seed": args.seed,
        "dataset_version": args.dataset_version, "checkpoint": best_dir,
        "num_beams": NUM_BEAMS, "n_examples": len(test_src), "metrics": scores,
    })

    config_summary = {
        "model_family": f"mamba_{args.size}", "base_model": f"scratch-mamba-{args.size}",
        "case_id": "X-0000", "seed": args.seed, "dataset_version": args.dataset_version,
        "output_dir": run_dir, "num_train_epochs": EPOCHS, "learning_rate": LR,
        **SIZES[args.size],
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config_summary, f, sort_keys=True)
    write_json(status_path, {"status": "completed", "case_id": "X-0000", "seed": args.seed,
                             "bleu": scores.get("bleu"), "comet": scores.get("comet")})

    print(f"[mamba_{args.size}] seed={args.seed}  BLEU={scores.get('bleu'):.2f}  "
          f"chrF2={scores.get('chrf2'):.2f}  COMET={scores.get('comet')}")


if __name__ == "__main__":
    main()
