# Copyright (c) OpenMMLab. All rights reserved.
from .dispatcher import FunctionDispatcher

paged_attention_fwd = FunctionDispatcher('paged_attention_fwd').make_caller()
paged_attention_fwd_prefill = FunctionDispatcher('paged_attention_fwd_prefill').make_caller()
