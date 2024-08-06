import torch

from vllm._C import ops
from torch.nn import functional as F

if __name__ == "__main__":
    position_ids = torch.range(0, 1235).reshape(1235, 1).cuda()
    query_states = torch.randn(1, 1235, 4096, dtype=torch.bfloat16).cuda()
    key_states = torch.randn(1, 1235, 4096, dtype=torch.bfloat16).cuda()
    cos_sin_cache = torch.randn(1, 1235, 128, dtype=torch.bfloat16).cuda().contiguous()
    head_dim = 128
    
    import pdb; pdb.set_trace()
    ops.rotary_embedding(position_ids,
                        query_states,
                        key_states,
                        head_dim,
                        cos_sin_cache,
                        True)

    import pdb; pdb.set_trace()
    # return query_states, key_states
