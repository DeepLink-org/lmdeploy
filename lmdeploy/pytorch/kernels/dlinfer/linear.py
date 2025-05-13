# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import dlinfer.ops as ext_ops
from torch import Tensor


def linear(x: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           dp_gather: bool = False,
           all_reduce: bool = False,
           rank: int = 0,
           tp_size: List[int] = None,
           group: str = ''):
    return ext_ops.linear(x,
                          weight,
                          bias=bias,
                          dp_gather=dp_gather,
                          all_reduce=all_reduce,
                          rank=rank,
                          tp_size=tp_size,
                          group=group)
