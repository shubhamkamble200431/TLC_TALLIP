"""
Differentiable length loss for TLC (brief section 20).

Audit of the TLC_multi implementation
-------------------------------------
`TLC_multi/training/loss.py::LengthAwareLoss` computed predicted length from
`logits.argmax(dim=-1)` (or a `predicted_ids` tensor that was never passed).
`argmax` has zero gradient almost everywhere, so that length term could not
train anything — and in practice it was dead code (`predicted_ids` was always
`None`, so the function returned plain NLL).

Replacement (this file)
-----------------------
A fully differentiable *expected-length* formulation:

    p_t_eos          = softmax(logits_t)[eos_id]                     # per decoder step
    expected_length  = sum_t (1 - p_t_eos)          over non-pad target steps
    length_loss      = |expected_length - target_length| / (target_length + eps)
    total_loss       = nll_loss + lambda_length * length_loss

Gradients flow  length_loss -> p_t_eos -> softmax -> decoder logits, so the
decoder is directly encouraged to place EOS mass consistent with the reference
length.  `target_length` is the number of non-pad reference tokens and carries
no gradient (it is data).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_length_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eos_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Differentiable expected generation length, per example.

    Parameters
    ----------
    logits : [B, T, V]
    labels : [B, T]   (ignore_index marks padding / non-target positions)

    Returns
    -------
    [B] float tensor of expected lengths.
    """
    probs = F.softmax(logits.float(), dim=-1)          # [B,T,V]
    p_eos = probs[..., eos_id]                          # [B,T]
    mask = (labels != ignore_index).float()            # [B,T]
    return ((1.0 - p_eos) * mask).sum(dim=1)           # [B]


class LengthAwareLoss(nn.Module):
    """
    `loss_type` selects the formula applied to the length discrepancy
    (ablation requested: huber / squared / off-by-one variants of the
    original L1-relative-error term):

      l1 (default)  rel_err = |L_hat - L| / (L + eps);            loss = mean(rel_err)
      squared       d = (L_hat - L) / (L + eps);                  loss = mean(d**2)
      huber         d = (L_hat - L) / (L + eps);                  loss = mean(huber(d, 0, delta=1.0))
                    (quadratic for |d|<=delta, linear beyond -- robust to outlier-length batches)
      l1_minus1     rel_err = |L_hat - (L-1)| / (L + eps);        loss = mean(rel_err)
                    (only the numerator's target shifts by 1, addressing the
                    EOS-position off-by-one bias documented in docs/TLC_MATH.txt
                    sec 3.4 -- the denominator stays L+eps, unchanged)
    """

    def __init__(
        self,
        eos_id: int,
        length_loss_lambda: float = 0.1,
        length_loss_epsilon: float = 1.0,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        use_length_loss: bool = True,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.eos_id = int(eos_id)
        self.lam = float(length_loss_lambda)
        self.eps = float(length_loss_epsilon)
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        self.use_length_loss = bool(use_length_loss)
        self.loss_type = str(loss_type)
        if self.loss_type not in ("l1", "squared", "huber", "l1_minus1"):
            raise ValueError(f"unknown length loss_type: {loss_type!r}")

    # ------------------------------------------------------------------
    def nll(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        vocab = logits.size(-1)
        return F.cross_entropy(
            logits.view(-1, vocab).float(),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def length_term(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_length is None:
            target_length = (labels != self.ignore_index).sum(dim=1)
        target_length = target_length.to(logits.device).float()
        exp_len = expected_length_from_logits(logits, labels, self.eos_id, self.ignore_index)
        denom = target_length + self.eps

        if self.loss_type == "l1":
            rel_err = (exp_len - target_length).abs() / denom
            return rel_err.mean()
        if self.loss_type == "l1_minus1":
            rel_err = (exp_len - (target_length - 1.0)).abs() / denom
            return rel_err.mean()
        d = (exp_len - target_length) / denom
        if self.loss_type == "squared":
            return (d ** 2).mean()
        # huber
        return F.huber_loss(d, torch.zeros_like(d), delta=1.0, reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.nll(logits, labels)
        if self.use_length_loss and self.lam > 0.0:
            loss = loss + self.lam * self.length_term(logits, labels, target_length)
        return loss
