# Copyright (c) OpenMMLab. All rights reserved.
from collections import deque
from dataclasses import dataclass
from typing import Deque

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
# from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger, logging_timer

# from ..messages import MessageStatus
from .scheduler import Scheduler, SchedulerOutput, SeqList

logger = get_logger('lmdeploy')


@dataclass
class PDSchedulerOutput(SchedulerOutput):
    blocks_to_migration = None


class PDScheduler(Scheduler):

    def __init__(self,
                 scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig,
                 is_prefill=True) -> None:
        super().__init__(scheduler_config, cache_config)
        self.waiting_for_migration: Deque[SeqList] = deque()
        self.is_prefill = is_prefill

    def schedule_blocks_to_migration(seq_list: SeqList):
        pass

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self, prealloc_size: int = 0):
        """schedule decoding."""
        if len(self.waiting_for_migration) != 0:
            migration_seq_list = self.waiting_for_migration.popleft()
            blocks_to_migration = self.schedule_blocks_to_migration(
                migration_seq_list)
        else:
            blocks_to_migration = None
        running, swap_in_map, swap_out_map, copy_map = super(
        )._schedule_decoding(prealloc_size)
        return PDSchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            copy_map=copy_map,
            blocks_to_migration=blocks_to_migration,
        )

    @logging_timer('SchedulePrefilling', logger)
    def _schedule_prefill(self):
        running, swap_in_map, swap_out_map, copy_map = super(
        )._schedule_prefill()
        return PDSchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            copy_map=copy_map,
        )

    def schedule(self, is_prefill: bool, prealloc_size: int = 0):
        """Schedule inputs for next steps."""
        if is_prefill:
            output = self._schedule_prefill()
        else:
            output = self._schedule_decoding(prealloc_size)

        return output

    def add_seq_to_waiting_for_migration(self, seq_list: SeqList):
        self.waiting_for_migration.append(seq_list)
