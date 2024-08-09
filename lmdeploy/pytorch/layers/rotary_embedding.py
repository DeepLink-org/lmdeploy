# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from ..backends import LayerType, get_backend
from ..backends.rotary_embedding import EmbeddingType


def build_rotary_embedding(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        emb_type: EmbeddingType = EmbeddingType.Default) -> nn.Module:
    """build rotary embedding op."""
    backend = get_backend()

    builder = backend.get_layer_impl_builder(LayerType.RotaryEmbedding)
    return builder.build(dim, max_position_embeddings, base, scaling_factor,
                         emb_type)


class ApplyRotaryEmb(nn.Module):
    """apply rotary embedding."""

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                seqlen: Tensor = None,
                inplace: bool = True):
        """forward."""
        return self.impl.forward(query, key, cos, sin, seqlen, inplace)
