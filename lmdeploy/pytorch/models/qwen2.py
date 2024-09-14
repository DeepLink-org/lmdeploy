# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from ..kernels import apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd
from ..weight_loader.dist_utils import (colwise_parallelize_linear,
                                        rowwise_parallelize_linear)


class PatchedQwen2Attention(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['o_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                cos, sin = self.rotary_emb(value_states,
                                           seq_len=max_kv_seq_length)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids,
                    context.position_ids_1d)
            return query_states, key_states, value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        query_states = query_states.view(-1, num_heads, head_dim)
        key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)

        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
        )

        attn_output = query_states

        use_sliding_windows = (getattr(self.config, 'sliding_window', None)
                               is not None and self.config.use_sliding_window)
        window_size = self.config.sliding_window
        if not use_sliding_windows:
            window_size = -1
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_seqlen=max_q_seq_length,
            window_size=window_size,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            world_size=world_size,
        )


class PatchedQwen2AttentionMuxi(nn.Module):

    def _load_weights(self, loader, rank: int, world_size: int,
                      device: torch.device):
        """load weights."""
        for mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)
        for mod_name in ['o_proj']:
            rowwise_parallelize_linear(getattr(self, mod_name),
                                       loader,
                                       rank=rank,
                                       world_size=world_size,
                                       prefix=mod_name)

    @classmethod
    def _distribute_output_fn(cls, outputs, **kwargs):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        context = self.context.context
        kv_seq_length = context.kv_seq_length
        q_seq_length = context.q_seq_length
        q_start_loc = context.q_start_loc
        block_offsets = context.block_offsets
        max_q_seq_length = context.max_q_seq_length
        max_kv_seq_length = context.max_kv_seq_length

        num_heads = self.num_heads // world_size
        num_kv_heads = self.num_key_value_heads // world_size
        head_dim = self.head_dim
        hidden_size = num_heads * head_dim

        def __qkv_proj(hidden_states):
            """qkv proj."""
            # query_states = self.q_proj(hidden_states)
            # key_states = self.k_proj(hidden_states)
            # value_states = self.v_proj(hidden_states)
            query_states = torch.matmul(hidden_states, self.q_proj.weight.data)
            if self.q_proj.bias is not None:
                query_states += self.q_proj.bias
            key_states = torch.matmul(hidden_states, self.k_proj.weight.data)
            if self.k_proj.bias is not None:
                key_states += self.k_proj.bias
            value_states = torch.matmul(hidden_states, self.v_proj.weight.data)
            if self.v_proj.bias is not None:
                value_states += self.v_proj.bias
            # import pdb; pdb.set_trace()
            return query_states, key_states, value_states

        def __rotary_emb_fn(query_states, key_states, value_states):
            if hasattr(self, 'rotary_emb'):
                # position_ids = context.position_ids_1d.unsqueeze(-1)
                cos, sin = self.rotary_emb(value_states,
                                           seq_len=max_kv_seq_length)

                new_cos = cos.squeeze(-2)
                new_cos = new_cos[..., :new_cos.shape[-1] // 2]
                new_sin = sin.squeeze(-2)
                new_sin = new_sin[..., :new_sin.shape[-1] // 2]
                cos_sin_cache = torch.cat((new_cos, new_sin), dim=-1)
                context.cos_sin_cache = cos_sin_cache

                # import pdb; pdb.set_trace()
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, 
                    key_states, 
                    context.position_ids_1d,
                    self.head_dim, 
                    context=context
                )
                # import pdb; pdb.set_trace()
            return  query_states.view(-1, num_heads, head_dim), key_states.view(-1, num_kv_heads, head_dim), value_states

        query_states, key_states, value_states = __qkv_proj(hidden_states)

        # query_states = query_states.view(-1, num_heads, head_dim)
        # key_states = key_states.view(-1, num_kv_heads, head_dim)
        value_states = value_states.view(-1, num_kv_heads, head_dim)

        query_states, key_states, value_states = __rotary_emb_fn(
            query_states, key_states, value_states)

        # import pdb; pdb.set_trace()
        fill_kv_cache(
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            q_start_loc,
            q_seq_length,
            kv_seq_length=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            block_offsets=block_offsets,
            context=context,
        )
        # import pdb; pdb.set_trace()

        attn_output = query_states

        use_sliding_windows = (getattr(self.config, 'sliding_window', None)
                               is not None and self.config.use_sliding_window)
        window_size = self.config.sliding_window
        if not use_sliding_windows:
            window_size = -1
        paged_attention_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            q_seqlens=q_seq_length,
            kv_seqlens=kv_seq_length,
            max_q_seq_length=max_q_seq_length,
            max_kv_seq_length=max_kv_seq_length,
            window_size=window_size,
            context=context,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          hidden_size)

        # attn_output = self.o_proj(attn_output)
        attn_output = torch.matmul(attn_output, self.o_proj.weight)
        if self.o_proj.bias:
            attn_output = attn_output + self.o_proj.bias

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of forward."""
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        return self._contiguous_batching_forward_impl(
            hidden_states,
            position_ids,
            past_key_value,
            world_size=world_size,
        )
