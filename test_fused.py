import torch

from vllm._C import ops
from torch.nn import functional as F

if __name__ == "__main__":
    seq_len = 47
    seq_len = 1021
    # seq_len = 1235
    position_ids = torch.arange(0, seq_len, dtype=torch.int64).reshape(seq_len, 1).cuda()
    query_states = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16).cuda()
    key_states = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16).cuda()
    cos_sin_cache = torch.randn(1, seq_len, 128, dtype=torch.bfloat16).cuda()
    head_dim = 128
    
    # import pdb; pdb.set_trace()
    ops.rotary_embedding(position_ids,
                        query_states,
                        key_states,
                        head_dim,
                        cos_sin_cache,
                        True)

    # import pdb; pdb.set_trace()

    print(query_states.shape, flush=True)
    print("Run done.", flush=True)
