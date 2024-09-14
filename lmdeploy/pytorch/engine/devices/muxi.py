# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base_device_utils import BaseDeviceUtils


class MUXIDeviceUtils(BaseDeviceUtils):

    device = 'muxi'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        _, _, block_size, _ = step_context.kv_caches[0][1].size()
        for i in range(step_context.q_start_loc.size(0)):
            single_attention_mask = torch.logical_not(
                torch.tril(
                    torch.ones(step_context.q_seq_length[i],
                               step_context.kv_seq_length[i],
                               dtype=torch.bool).cuda(),
                    diagonal=step_context.kv_seq_length[i] -
                    step_context.q_seq_length[i],
                ))
            attention_mask.append(single_attention_mask)
            history_length = step_context.history_lengths[i]
            block_idx = history_length // block_size
            block_loc = step_context.block_offsets[i][block_idx]
            token_loc = history_length % block_size
            for _ in range(step_context.q_seq_length[i]):
                kv_start_indices.append(block_loc * block_size + token_loc)
                if _ == step_context.q_seq_length[i] - 1:
                    break
                token_loc = (token_loc + 1) % block_size
                block_idx = block_idx if token_loc else block_idx + 1
                block_loc = step_context.block_offsets[i][block_idx]
        kv_start_indices = torch.tensor(
            kv_start_indices, device=step_context.block_offsets.device)
        #attention_mask = torch.stack(attention_mask)
        setattr(step_context, 'kv_start_indices', kv_start_indices)
        # setattr(step_context, 'attention_mask', attention_mask)

        def _make_cu_seqlens(seqlens):
            cu_seqlens = seqlens.cumsum(0)
            cu_zero = cu_seqlens.new_zeros(1)
            cu_seqlens = torch.cat([cu_zero, cu_seqlens])
            return cu_seqlens

        cu_seqlens_q = _make_cu_seqlens(step_context.q_seq_length).int()
        cu_seqlens_kv = _make_cu_seqlens(step_context.kv_seq_length).int()

        setattr(step_context, 'cu_seqlens_q', cu_seqlens_q)
        setattr(step_context, 'cu_seqlens_kv', cu_seqlens_kv)

        step_context.block_offsets = step_context.block_offsets.to(torch.int32)
        step_context.kv_seq_length = step_context.kv_seq_length.to(torch.int32)

        return step_context
