# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

fused_rotary_emb = FunctionDispatcher('fused_rotary_emb').make_caller()
fused_rotary_emb_op = FunctionDispatcher('fused_rotary_emb_op').make_caller()
fused_rotary_emb_eager = FunctionDispatcher('fused_rotary_emb_eager').make_caller()
