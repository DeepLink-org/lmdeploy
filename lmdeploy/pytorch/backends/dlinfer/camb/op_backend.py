# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ...base import OpType
from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')


class CambOpsBackend(DlinferOpsBackend):
    """camb layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'camb'

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            num_heads,
            block_size,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        return (
            num_heads,
            block_size,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        _, _, block_size, _ = step_context.kv_caches[0][0].shape
        device = step_context.block_offsets.device
        batch_size = step_context.q_start_loc.shape[0]

        is_unpaged_prefill = False
        q_start_loc = step_context.q_start_loc
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens.to(torch.int32)
        max_q_seq_len = torch.max(q_seqlens).cpu().item()
        max_kv_seq_len = torch.max(kv_seqlens).cpu().item()

        cu_seqlens = torch.zeros(batch_size+1, dtype=torch.int32, device=device)
        cu_seqlens[:-1] = step_context.q_start_loc
        cu_seqlens[-1] = step_context.q_seqlens.sum()

        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())

        for i in range(batch_size):
            q_seq_len = int(step_context.q_seqlens[i])
            kv_seq_len = int(step_context.kv_seqlens[i])
            history_length = kv_seq_len - q_seq_len
            block_idx = history_length // block_size
            block_loc = step_context.block_offsets[i][block_idx]
            token_loc = history_length % block_size
            for j in range(q_seq_len):
                kv_start_indices.append(block_loc * block_size + token_loc)
                if j == q_seq_len - 1:
                    break
                token_loc = (token_loc + 1) % block_size
                block_idx = block_idx if token_loc else block_idx + 1
                block_loc = step_context.block_offsets[i][block_idx]
        kv_start_indices = torch.tensor(kv_start_indices, device=device, dtype=torch.int32)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets.to(torch.int32),
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=None,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            cu_seqlens=cu_seqlens,
            is_flash_attn_support_inplace=False,
            is_mock_q_start_loc=True,
        )

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from .graph_runner import CAMBGraphRunner
        return CAMBGraphRunner(model, model_config, cache_config,
                               backend_config, device)