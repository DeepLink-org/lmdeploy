# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from lmdeploy.pytorch.kernels.camb import apply_rotary_pos_emb

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl
from .attention import CambAttentionMetadata

class CambApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """camb Apply rotary embedding implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                attn_metadata: CambAttentionMetadata,
                inplace: bool = True):
        """forward."""
        cos_sin_ids = attn_metadata.cos_sin_ids
        cu_seqlens = attn_metadata.cu_seqlens

        if inplace:
            q_embed = None
            k_embed = None
        else:
            q_embed = torch.empty_like(query)
            k_embed = torch.empty_like(key)
        return apply_rotary_pos_emb(query, key, cos, sin, q_embed, k_embed, cos_sin_ids, cu_seqlens)


class CambApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """camb Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """build implementation."""
        return CambApplyRotaryEmbImpl()

