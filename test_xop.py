import torch
import xformers.ops as xops

if __name__ == "__main__":
    q = torch.randn(1, 1226, 16, 112, dtype=torch.float16)
    k = torch.randn(1, 1226, 16, 112, dtype=torch.float16)
    v = torch.randn(1, 1226, 16, 112, dtype=torch.float16)
    scale = 0.94

    import pdb; pdb.set_trace()

    out = xops.memory_efficient_attention(
        q, k, v, scale=scale,
    )

    import pdb; pdb.set_trace()

    
