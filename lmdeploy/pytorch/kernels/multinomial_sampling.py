# Copyright (c) OpenMMLab. All rights reserved.
import torch


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    import pdb;pdb.set_trace()
    sampled_index = torch.multinomial(scores, num_samples=1, replacement=True)
    outputs = torch.gather(indices, dim=1, index=sampled_index)
    return outputs.view(-1)
