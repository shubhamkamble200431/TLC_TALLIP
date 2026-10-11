"""
Optional positional-encoding interventions (C = RoPE, D = ALiBi).

RoPE
----
Ported from TLC_multi/models/positional_encodings.py: a per-attention-layer
RotaryEmbedding wraps the module `forward`, rotating q/k before the original
attention computation.  Applied to every self-attention module that exposes
`q_proj` (encoder + decoder).

ALiBi
-----
TLC_multi only *attached* an unused `alibi_bias` buffer to each layer (a no-op,
which its own log message admitted).  Brief section 20 forbids silently keeping
an invalid implementation, so ALiBi here is actually applied: a linear-bias
term  -slope_h * |i - j|  is added, per head, into the additive attention mask
of every *self-attention* module.  This is the standard ALiBi formulation for a
bidirectional encoder; the decoder's causal mask is preserved because the bias
is *added* to (not replacing) the existing mask.  Cross-attention is left
untouched (ALiBi is not defined for it).

Both patches are pure-Python module wrappers — no change to model weights, so
checkpoints load back into a vanilla HF model for evaluation.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================================
# RoPE
# =========================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device=None):
        device = device or self.inv_freq.device
        self.inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(self, q, k):
        device = q.device
        need = max(q.shape[2], k.shape[2])
        hd = q.shape[-1]
        if need > self.cos_cache.shape[2] or self.cos_cache.device != device:
            self._build_cache(need, device=device)
        return (
            _apply_rot(q, self.cos_cache[:, :, : q.shape[2], :hd], self.sin_cache[:, :, : q.shape[2], :hd]),
            _apply_rot(k, self.cos_cache[:, :, : k.shape[2], :hd], self.sin_cache[:, :, : k.shape[2], :hd]),
        )


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rot(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


def apply_rope_to_model(model: nn.Module, head_dim: int, max_seq_len: int = 2048) -> int:
    patched = 0
    for _, module in model.named_modules():
        if "Attention" in type(module).__name__ and hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            _patch_attn_rope(module, RotaryEmbedding(head_dim, max_seq_len))
            patched += 1
    logger.info("[RoPE] patched %d attention layers (head_dim=%d)", patched, head_dim)
    return patched


def _patch_attn_rope(attn: nn.Module, rope: RotaryEmbedding) -> None:
    num_heads = attn.num_heads
    head_dim = rope.dim
    q_lin, k_lin = attn.q_proj, attn.k_proj
    object.__setattr__(attn, "_rope", rope)
    orig_forward = attn.forward

    class _Fixed(nn.Module):
        def __init__(self, out):
            super().__init__()
            self._out = out

        def forward(self, _x):
            return self._out

    def rope_forward(hidden_states, key_value_states=None, past_key_value=None,
                     attention_mask=None, layer_head_mask=None, output_attentions=False, **kw):
        kv = key_value_states if key_value_states is not None else hidden_states
        raw_q = F.linear(hidden_states, q_lin.weight, q_lin.bias)
        raw_k = F.linear(kv, k_lin.weight, k_lin.bias)
        Bq, Sq, _ = raw_q.shape
        Bk, Sk, _ = raw_k.shape
        q4 = raw_q.view(Bq, Sq, num_heads, head_dim).transpose(1, 2)
        k4 = raw_k.view(Bk, Sk, num_heads, head_dim).transpose(1, 2)
        q_rot, k_rot = attn._rope(q4, k4)
        q_flat = q_rot.transpose(1, 2).contiguous().view(Bq, Sq, num_heads * head_dim)
        k_flat = k_rot.transpose(1, 2).contiguous().view(Bk, Sk, num_heads * head_dim)
        sq, sk = attn.q_proj, attn.k_proj
        attn.q_proj, attn.k_proj = _Fixed(q_flat), _Fixed(k_flat)
        try:
            return orig_forward(
                hidden_states, key_value_states=key_value_states, past_key_value=past_key_value,
                attention_mask=attention_mask, layer_head_mask=layer_head_mask,
                output_attentions=output_attentions, **kw,
            )
        finally:
            attn.q_proj, attn.k_proj = sq, sk

    attn.forward = rope_forward


# =========================================================================
# ALiBi
# =========================================================================
def _alibi_slopes(n_heads: int) -> torch.Tensor:
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(pow2(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    base = pow2(closest)
    extra = pow2(2 * closest)[::2][: n_heads - closest]
    return torch.tensor(base + extra)


class ALiBiBias(nn.Module):
    """Symmetric linear-distance bias  b[h,i,j] = -slope_h * |i - j|."""

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        slopes = _alibi_slopes(num_heads)  # [H]
        pos = torch.arange(max_seq_len)
        dist = (pos[None, :] - pos[:, None]).abs().float()  # [S,S]
        bias = -slopes[:, None, None] * dist[None, :, :]  # [H,S,S]
        self.register_buffer("bias", bias.unsqueeze(0), persistent=False)  # [1,H,S,S]

    def get(self, q_len: int, k_len: int, device, dtype):
        if k_len > self.bias.shape[-1] or q_len > self.bias.shape[-2]:
            self._extend(max(q_len, k_len))
        return self.bias[:, :, :q_len, :k_len].to(device=device, dtype=dtype)

    def _extend(self, seq_len: int):
        slopes = _alibi_slopes(self.num_heads)
        pos = torch.arange(seq_len)
        dist = (pos[None, :] - pos[:, None]).abs().float()
        bias = -slopes[:, None, None] * dist[None, :, :]
        self.register_buffer("bias", bias.unsqueeze(0), persistent=False)


def apply_alibi_to_model(model: nn.Module, num_heads: int, max_seq_len: int = 2048) -> int:
    """
    Wrap every *self-attention* module so the ALiBi bias is folded into the
    additive `attention_mask` argument (broadcast over heads).  Preserves any
    existing (padding / causal) mask because the bias is summed with it.
    """
    shared = ALiBiBias(num_heads, max_seq_len)
    patched = 0
    for name, module in model.named_modules():
        is_attn = "Attention" in type(module).__name__ and hasattr(module, "q_proj")
        is_self = name.endswith("self_attn") or name.endswith("self_attention")
        if is_attn and is_self:
            _patch_attn_alibi(module, shared)
            patched += 1
    logger.info("[ALiBi] applied additive bias to %d self-attention layers (heads=%d)", patched, num_heads)
    return patched


def _patch_attn_alibi(attn: nn.Module, alibi: ALiBiBias) -> None:
    object.__setattr__(attn, "_alibi", alibi)
    orig_forward = attn.forward

    def alibi_forward(hidden_states, key_value_states=None, past_key_value=None,
                      attention_mask=None, layer_head_mask=None, output_attentions=False, **kw):
        # self-attention only: q_len == k_len == sequence length of hidden_states
        q_len = hidden_states.shape[1]
        kv = key_value_states if key_value_states is not None else hidden_states
        k_len = kv.shape[1]
        bias = attn._alibi.get(q_len, k_len, hidden_states.device, hidden_states.dtype)  # [1,H,q,k]
        if attention_mask is None:
            new_mask = bias.expand(hidden_states.shape[0], attn._alibi.num_heads, q_len, k_len).contiguous()
        else:
            # attention_mask: [B,1,q,k] (float additive) -> broadcast-add the [1,H,q,k] bias.
            # builder.py forces eager attention when ALiBi is on, so a broadcast
            # (non-contiguous) mask is fine here.
            new_mask = attention_mask + bias
        return orig_forward(
            hidden_states, key_value_states=key_value_states, past_key_value=past_key_value,
            attention_mask=new_mask, layer_head_mask=layer_head_mask,
            output_attentions=output_attentions, **kw,
        )

    attn.forward = alibi_forward
