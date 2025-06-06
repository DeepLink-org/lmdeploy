# Copyright (c) OpenMMLab. All rights reserved.
from ..default.token_dispatcher import AlltoAllTokenDispatcher
from ..token_dispatcher import TokenDispatcherImpl


class TokenDispatcherBuilder:
    """token dispatcher builder."""

    @staticmethod
    def build(
        group,
        num_experts,
        num_local_experts,
        hidden_size = 0,
        params_dtype = None,
    ) -> TokenDispatcherImpl:
        """build."""
        return AlltoAllTokenDispatcher(
            group,
            num_experts,
            num_local_experts,
        )
