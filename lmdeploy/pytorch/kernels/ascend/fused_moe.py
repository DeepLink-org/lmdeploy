import torch
from torch import Tensor


def fused_moe(hidden_states: torch.Tensor,  gate_up_weights: torch.Tensor,
              down_weights: torch.Tensor, topk_weights: torch.Tensor,
              topk_ids: torch.LongTensor, topk: int=1):
    """forward."""
    seq_length = hidden_states.size(0)
    moe_output = torch.zeros_like(hidden_states)

    for i in range(seq_length):
        current_hidden_state = hidden_states[i]

        # faster than remove the for loop
        for j in range(topk):
            import pdb;pdb.set_trace()
            expert_id = topk_ids[i][j]
            weight = topk_weights[i][j]

            up_weight = gate_up_weights[expert_id]
            up_proj = torch.matmul(up_weight, current_hidden_state)

            gate_cache, up_cache = up_proj.chunk(2, -1)
            gate_cache = torch.nn.functional.silu(gate_cache,
                                                    inplace=True) * up_cache

            down_weight = down_weights[expert_id]
            down_proj = torch.matmul(down_weight, gate_cache)

            moe_output[i] += weight * down_proj

    return moe_output
