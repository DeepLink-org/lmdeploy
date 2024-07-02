import torch
from torch import Tensor
import vllm._C as vllm_ops


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    output = torch.empty_like(hidden_states)
    vllm_ops.ops.rms_norm(output, hidden_states, weight, eps)
    return output
