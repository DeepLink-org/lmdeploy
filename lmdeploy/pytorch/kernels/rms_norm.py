import torch
from torch import Tensor
#import deeplink_ext.cpp_extensions as ext

def torch_forward(hidden_states, weight, eps=1e-6):
    """pytorch forward."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance +
                                                eps)
    return weight * hidden_states.to(input_dtype)

rms_norm = torch_forward



#def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
#    output = torch.empty_like(hidden_states)
#    inv_rms_shape = list(hidden_states.shape[:-1]) + [1]
#    inv_rms = torch.empty(inv_rms_shape, dtype=torch.float32, device=hidden_states.device)
#    ext.rms_norm(output, inv_rms, hidden_states, weight.shape, weight, None, eps)
#    return output


if __name__ == '__main__':
    import time

    def torch_forward(hidden_states, weight, variance_epsilon=1e-6):
        """pytorch forward."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    variance_epsilon)
        return weight * hidden_states.to(input_dtype)

    def test_rms_norm(bsz, ctx_len, feat_len, dtype):
        """test rms norm."""
        input = torch.empty((bsz, ctx_len, feat_len),
                            dtype=dtype,
                            device='cuda').normal_(mean=0.,
                                                   std=0.5).contiguous()
        weight = torch.empty((feat_len), dtype=dtype,
                             device='cuda').normal_(mean=0.,
                                                    std=0.5).contiguous()
        triton_output = rms_norm(hidden_states=input, weight=weight)
        torch_output = torch_forward(hidden_states=input, weight=weight)
        assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)

        N_REPEATS = 20

        t0 = time.time()
        for _ in range(N_REPEATS):
            torch_forward(hidden_states=input, weight=weight)

        t1 = time.time()
        for _ in range(N_REPEATS):
            rms_norm(hidden_states=input, weight=weight)
        t2 = time.time()

        torch_cost = (t1 - t0) / N_REPEATS * 1000
        triton_cost = (t2 - t1) / N_REPEATS * 1000
        print(
            'input {} weight {} dtype {}\n  torch {:.3f} triton {:.3f} (ms)\n'.
            format(input.shape, weight.shape, dtype, torch_cost, triton_cost))

    test_rms_norm(1, 8128, 5120, torch.float16)
    test_rms_norm(1, 8128, 5120, torch.float32)
    test_rms_norm(1, 992, 128, torch.float16)
    test_rms_norm(1, 65537, 128, torch.float32)
