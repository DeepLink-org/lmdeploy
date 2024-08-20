import math
import torch

import vllm._C.ops as ops
import xformers.ops as xops

from torch.nn import functional as F

# from flash_attn import flash_attn_varlen_func

from maca_extension import ops as ext_ops

def dump_tensor(x, name):
    import pickle
    with open(f'/home/pujiang/zhousl/{name}.pkl', 'wb') as f:
        if isinstance(x, torch.Tensor):
                pickle.dump(x.cpu(), f)
        else:
            pickle.dump(x, f)

# def load_tensor(name):
#     import pickle
#     with open(f'/home/pujiang/zhousl/{name}.pkl', 'rb') as f:
#         x = pickle.load(f)
#     if isinstance(x, torch.Tensor):
#         return x.cuda()
#     return x
            
def load_tensor(name):
    import pickle
    with open(f'/home/pujiang/zhousl/tmp/{name}.pkl', 'rb') as f:
        x = pickle.load(f)
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x


if __name__ == "__main__":
    query = torch.randn(1, 16, 32, 258, dtype=torch.float16).cuda().contiguous()
    key = torch.randn(1, 16, 32, 258, dtype=torch.float16).cuda().contiguous()
    value = torch.randn(1, 16, 32, 258, dtype=torch.float16).cuda().contiguous()
    scale = 0.94
    win_size = (-1, -1)
    _, seq_len_q, H, D = query.shape
    cu_seqlens_q = torch.tensor([0, seq_len_q], dtype=torch.int32, device=query.device)

    query_states = load_tensor("query_states")
    key_states = load_tensor("key_states")
    value_states = load_tensor("value_states")
    key_cache = load_tensor("key_cache")
    value_cache = load_tensor("value_cache")
    attn_output = load_tensor("attn_output")
    block_offsets = load_tensor("block_offsets")
    q_seq_length = load_tensor("q_seq_length")
    kv_seq_length = load_tensor("kv_seq_length")
    cu_seqlens_q = load_tensor("cu_seqlens_q")
    cu_seqlens_kv = load_tensor("cu_seqlens_kv")

    _, head, dim = key_states.size()
    max_q_seq_length = query_states.shape[0]
    max_kv_seq_length = key_states.shape[0]
    window_size=(-1, -1)
    return_softmax=False
    softmax_scale=float(1 / math.sqrt(dim))

    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    query_states, key_states, value_states = [maybe_contiguous(x) for x in (query_states, key_states, value_states)]

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = ext_ops.flash_attn_varlen_fwd(
        query_states,
        key_states,
        value_states,
        None,
        cu_seqlens_q,
        cu_seqlens_kv,
        None,
        None,
        max_q_seq_length,
        max_kv_seq_length,
        0.0,
        softmax_scale,
        False,
        True,
        window_size[0],
        window_size[1],
        return_softmax,
        None,
    )
    
    import pdb; pdb.set_trace()

    out = flash_attn_varlen_func(query.view(-1, H, D), 
                                 key.view(-1, H, D), 
                                 value.view(-1, H, D),
                                 cu_seqlens_q=cu_seqlens_q,
                                 cu_seqlens_k=cu_seqlens_q,
                                 max_seqlen_q=seq_len_q,
                                 max_seqlen_k=seq_len_q,
                                 softmax_scale=scale, 
                                 window_size=win_size)
    
    import pdb; pdb.set_trace()

    query_states = load_tensor("query_states")
    key_cache = load_tensor("key_cache")
    value_cache = load_tensor("value_cache")
    block_offsets = load_tensor("block_offsets")
    kv_seqlens = load_tensor("kv_seqlens")

    max_kv_seq_length = kv_seqlens.item()
    block_num, head, block_size, dim = value_cache.size()

    # import pdb; pdb.set_trace()

    attn_output = query_states
    ext_ops.paged_attention_v1(
        attn_output,
        query_states,
        key_cache,
        value_cache,
        head,
        float(1 / math.sqrt(dim)), # scale
        block_offsets,
        kv_seqlens,
        block_size,
        max_kv_seq_length,
        None,
        'auto',
    )

    print("Run done1.", flush=True)
