"""
Dedicated embedding for the 5 LEN_BUCKET_* length-control tokens (ablation:
does the length signal benefit from more/less representational capacity,
independent of the model's frozen pretrained d_model?).

Default behaviour (bucket_embed_dim=None) leaves the 5 tokens as ordinary
rows of the model's own tied vocab embedding -- ``src/models/builder.py``'s
``tokenizer.add_special_tokens`` + ``model.resize_token_embeddings`` already
does this, see docs/TLC_MATH.txt sec 2/6.

When ``bucket_embed_dim`` IS set, this wraps the decoder's input embedding
module so that ONLY the 5 bucket-token positions bypass the tied vocab
embedding entirely and instead go through:

    nn.Embedding(5, bucket_embed_dim)  ->  nn.Linear(bucket_embed_dim, d_model)

a small dedicated lookup + learned projection, swept over bucket_embed_dim in
{512, 1024, 2048, 4096} independently of NLLB/mBART's fixed d_model=1024 --
the whole point of this ablation. Every other token position is untouched
(falls through to the original embedding unchanged).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BucketAwareEmbedding(nn.Module):
    def __init__(self, base_embedding: nn.Embedding, bucket_token_ids: List[int],
                bucket_embed_dim: int, d_model: int):
        super().__init__()
        self.base = base_embedding
        self.register_buffer("bucket_token_ids", torch.tensor(bucket_token_ids, dtype=torch.long),
                             persistent=False)
        self.bucket_embed = nn.Embedding(len(bucket_token_ids), bucket_embed_dim)
        self.proj = nn.Linear(bucket_embed_dim, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        ids = self.bucket_token_ids.to(input_ids.device)
        # [B,T] -> which of the 5 bucket slots each position matches, or -1 if none
        match = (input_ids.unsqueeze(-1) == ids.view(1, 1, -1))   # [B,T,5]
        is_bucket = match.any(dim=-1)                              # [B,T]
        if not bool(is_bucket.any()):
            return out
        slot = match.float().argmax(dim=-1)                        # [B,T] (garbage where not is_bucket)
        bucket_vecs = self.proj(self.bucket_embed(slot))           # [B,T,d_model]
        return torch.where(is_bucket.unsqueeze(-1), bucket_vecs.to(out.dtype), out)


def wrap_bucket_embedding(model, bucket_token_ids: List[int], bucket_embed_dim: int) -> None:
    """In place: replace `model.get_decoder().embed_tokens` with a
    BucketAwareEmbedding wrapping the original. NLLB (M2M100) and mBART-50
    both expose this exact attribute path."""
    decoder = model.get_decoder()
    d_model = model.config.d_model
    decoder.embed_tokens = BucketAwareEmbedding(
        decoder.embed_tokens, bucket_token_ids, bucket_embed_dim, d_model
    )
