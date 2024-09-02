import torch
from torch import Tensor
import dlinfer.ops as ext_ops

from vllm._C import ops


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6, residual: torch.Tensor = None):
    if residual is not None:
        ext_ops.add_rms_norm(
            hidden_states,
            residual,
            weight,
            eps,
        )
        return hidden_states, residual
    else:
        output = ext_ops.rms_norm(hidden_states, weight, eps)
        return output
