# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
# from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger, logging_timer

from ..messages import LogicalTokenBlocks, MessageStatus, SchedulerSequence
# from ..messages import MessageStatus
from .scheduler import Scheduler, SchedulerOutput, SeqList

logger = get_logger('lmdeploy')


@dataclass
class PDMigrationMeta:
    p_blocks_to_migration: List[slice] = field(default_factory=list)
    d_blocks_to_migration: List[slice] = field(default_factory=list)
    seq_list: List[SchedulerSequence] = field(default_factory=list)
    src_model_agent_id: int = -1
    dst_model_agent_id: int = -1


@dataclass
class PDSchedulerOutput(SchedulerOutput):
    migration_meta: PDMigrationMeta = None


class PDScheduler(Scheduler):

    def __init__(self, scheduler_config: SchedulerConfig,
                 cache_config: CacheConfig) -> None:
        super().__init__(scheduler_config, cache_config)
        self.migration_info_dict: Dict[Tuple[int, int], List[Tuple[
            SchedulerSequence, np.ndarray, np.ndarray]]] = defaultdict(list)

    @property
    def waiting_for_migration(self):
        """get waiting for migrating sequence."""
        seq_map = self.seq_manager.get_sequences(
            MessageStatus.WAITING_FOR_MIGRATION)
        return list(seq_map.values())

    def has_waiting_for_migration(self):
        return self.seq_manager.num_sequences(
            MessageStatus.WAITING_FOR_MIGRATION) > 0

    def has_migration_done(self):
        return self.seq_manager.num_sequences(MessageStatus.MIGRATION_DONE) > 0

    @property
    def migration_done(self):
        """get migration done."""
        seq_map = self.seq_manager.get_sequences(MessageStatus.MIGRATION_DONE)
        return list(seq_map.values())

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_migration(self):
        migration_meta = self.collect_migration_meta()
        return PDSchedulerOutput(running=list(),
                                 swap_in_map=dict(),
                                 swap_out_map=dict(),
                                 copy_map=dict(),
                                 migration_meta=migration_meta)

    @logging_timer('ScheduleDecoding', logger)
    def _schedule_decoding(self, prealloc_size: int = 0):
        """schedule decoding."""
        migration_done = self.migration_done
        self.change_msg_status(migration_done, MessageStatus.RUNNING)
        running, swap_in_map, swap_out_map, copy_map = super(
        )._schedule_decoding(prealloc_size)
        return PDSchedulerOutput(
            running=running,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            copy_map=copy_map,
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
            # print("_schedule_prefill")
            output = self._schedule_prefill()
        elif len(self.waiting_for_migration) != 0:
            # print("_schedule_migration")
            output = self._schedule_migration()
        else:
            # print("_schedule_decoding")
            output = self._schedule_decoding(prealloc_size)

        return output

    def remove_sessions_for_seqs(self, seq_list: SeqList):
        for seq in seq_list:
            self.sessions.pop(seq.seq_id)

    def change_msg_status(self, seq_list: SeqList, status: MessageStatus):
        for seq in seq_list:
            self._set_message_status(seq, status)

    def update_migration_info(self, seq_list: SeqList,
                              p_blocks_list: List[np.ndarray], src_id: int,
                              dst_id: int):
        for seq, p_blocks in zip(seq_list, p_blocks_list):
            # transfer session from P to D and add seq to seq_manager in D
            self._set_message_status(seq, MessageStatus.WAITING_FOR_MIGRATION)
            seq.seq_manager.remove_sequence(seq)
            seq.session.seq_manager = self.seq_manager
            self.sessions[seq.seq_id] = seq.session
            self.seq_manager.add_sequence(seq)
            # allocate new logical blocks for seq
            d_blocks = self.block_manager.allocator.allocate(
                len(seq.logical_blocks), 'gpu')
            seq.logical_blocks = LogicalTokenBlocks(d_blocks)
            # update migration_info_dict:
            self.migration_info_dict[(src_id, dst_id)].append(
                (seq, p_blocks, d_blocks))

    def collect_migration_meta(self) -> PDMigrationMeta:
        migration_meta = PDMigrationMeta()
        (src_id,
         dst_id), seq_meta_list = list(self.migration_info_dict.items())[0]
        migration_meta.src_model_agent_id = src_id
        migration_meta.dst_model_agent_id = dst_id
        p_blocks_to_migration = []
        d_blocks_to_migration = []
        for seq, p_blocks, d_blocks in seq_meta_list:
            p_blocks_to_migration += p_blocks.tolist()
            d_blocks_to_migration += d_blocks.tolist()
            migration_meta.seq_list.append(seq)
        # TODO different seqs may require different p_blocks slice
        # # and corresponding d_blocks slice
        migration_meta.p_blocks_to_migration = find_slices(
            p_blocks_to_migration)
        migration_meta.d_blocks_to_migration = find_slices(
            d_blocks_to_migration)
        del self.migration_info_dict[(src_id, dst_id)]
        return migration_meta


def find_slices(array):
    slices = []
    sorted_array = sorted(array)
    start = sorted_array[0]
    for i in range(1, len(sorted_array)):
        if sorted_array[i] != sorted_array[i - 1] + 1:
            slices.append(slice(start, sorted_array[i - 1] + 1))
            start = sorted_array[i]
    slices.append(slice(start, sorted_array[-1] + 1))
    return slices
