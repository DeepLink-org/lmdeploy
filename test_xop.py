import torch
import xformers.ops as xops

from torch.nn import functional as F

if __name__ == "__main__":
    query = torch.randn(1, 1226, 16, 112, dtype=torch.float16).cuda().contiguous()
    key = torch.randn(1, 1226, 16, 112, dtype=torch.float16).cuda().contiguous()
    value = torch.randn(1, 1226, 16, 112, dtype=torch.float16).cuda().contiguous()
    scale = 0.94

    import pdb; pdb.set_trace()

    res1 = xops.memory_efficient_attention(
        query, key, value, scale=scale,
    )

    import pdb; pdb.set_trace()

    print(res1.shape, flush=True)
    print("Run done1.", flush=True)

    # Equivalent pytorch code memory_efficient_attention:
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = query @ key.transpose(-2, -1)
    attn = attn.softmax(-1)
    attn = F.dropout(attn, 0.0)
    res2 = attn @ value

    import pdb; pdb.set_trace()

    print(res2.shape, flush=True)
    print("Run done2.", flush=True)