# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from typing import Any, Dict, List, Tuple

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.engine.model_agent import build_model_agent
from lmdeploy.utils import get_logger

from .base import ExecutorBase
from .dist_utils import find_available_port, init_process_group, setup_master_addr

logger = get_logger('lmdeploy')

PG_WAIT_TIMEOUT = 1800


def get_device_str():
    """get device str."""
    device_type = get_device_manager().current_context().device_type

    if device_type == 'cuda':
        device_type = 'GPU'

    return device_type


def _wait_until_pg_ready(current_placement_group: 'PlacementGroup'):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is not created within time.
    """
    # copy from vLLM
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            'Waiting for creating a placement group of specs for '
            '%d seconds. specs=%s. Check '
            '`ray status` to see if you have enough resources,'
            ' and make sure the IP addresses used by ray cluster'
            ' are the same as VLLM_HOST_IP environment variable'
            ' specified in each node if you are running on a multi-node.', int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError('Cannot provide a placement group of '
                         f'{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See '
                         '`ray status` to make sure the cluster has enough resources.') from None


def init_ray_cluster(world_size: int, ray_address: str = None):
    """init ray cluster."""
    # modifier from vLLM
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    device_str = get_device_str()

    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if not current_placement_group:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if world_size > num_devices_in_cluster:
            logger.warning(
                'The number of required %ss exceeds the total '
                'number of available %ss in the placement group.', device_str, device_str)
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{device_str: 1.0} for _ in range(world_size)])

        # gcs_addr = ray.get_runtime_context().gcs_address
        # master_addr = gcs_addr.split(':')[0]
        # current_ip = master_addr
        # # This way, at least bundle is required to be created in a current
        # # node.
        # placement_group_specs[0][f'node:{current_ip}'] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(placement_group_specs, strategy='PACK')
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    # Set the placement group in the parallel config
    placement_group = current_placement_group
    return placement_group


def _get_master_addr():
    gcs_addr = ray.get_runtime_context().gcs_address
    master_addr = gcs_addr.split(':')[0]
    return master_addr


class RayWorkerWrapper:
    """worker warpper."""

    def __init__(
        self,
        model_path: str,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        dp: int,
        tp: int,
        adapters: Dict[str, str] = None,
        device_type: str = 'cuda',
        dtype: str = 'auto',
        log_level: int = 30,
        nproc_per_node: int = None,
    ):
        from lmdeploy.tokenizer import Tokenizer
        self.model_path = model_path
        self.model_config = ModelConfig.from_pretrained(model_path, dtype=dtype, tp=tp)
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.tokenizer = Tokenizer(model_path).model.model
        self.adapters = adapters
        self.device_type = device_type
        self.log_level = log_level
        self.dp = dp
        self.tp = tp
        self.world_size = tp * dp
        if nproc_per_node is None:
            nproc_per_node = self.world_size
        self.nproc_per_node = nproc_per_node
        self.device_type = device_type
        self.node_ip = ray.util.get_node_ip_address()

        logger.setLevel(log_level)
        self.model = None

    def get_node_ip(self):
        """get worker ip."""
        return self.node_ip

    def init_process_group(self, rank: int, master_addr: str, master_port: str):
        """initialize process group."""
        setup_master_addr(master_addr, master_port)
        self.rank = rank
        self.device_id = rank % self.nproc_per_node

        init_process_group(rank, self.world_size, self.nproc_per_node)

    def build_model(self):
        """build model."""
        self.dist_ctx = DistContext.build(self.rank, self.tp, self.dp, self.nproc_per_node)
        self.device_ctx = DeviceContext(device_type=self.device_type)

        self.model_agent = build_model_agent(model_path=self.model_path,
                                             model_config=self.model_config,
                                             cache_config=self.cache_config,
                                             backend_config=self.backend_config,
                                             tokenizer=self.tokenizer,
                                             device_ctx=self.device_ctx,
                                             dist_ctx=self.dist_ctx,
                                             adapters=self.adapters)
        self.model_agent.build_model()

    def gather_free_mem(self):
        """gather free mem."""
        return self.model_agent.get_free_mem()

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.model_agent.set_cache_config(cache_config)

    def set_model_config(self, model_config: ModelConfig):
        """set all model config."""
        self.model_agent.set_model_config(model_config)

    def build_graph_runner(self):
        """build graph runner."""
        self.model_agent.build_graph_runner()

    def build_cache_engine(self):
        """build cache engine."""
        self.model_agent.build_cache_engine()

    def get_input_processor(self):
        """build cache engine."""
        return self.model_agent.get_input_processor()

    def start(self):
        """start engine loop."""
        self.model_agent.start()

    def stop(self):
        """stop engine loop."""
        self.model_agent.stop()

    async def forward_async(self, inputs):
        """start forward."""
        self.model_agent.set_forward_inputs(inputs)

    async def get_output_async(self):
        """get output async."""
        return await self.model_agent.get_output_async()

    def release(self):
        """stop engine loop."""
        self.model_agent.release()


class RayExecutor(ExecutorBase):
    """ray executor."""

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 tokenizer: Any,
                 dp: int,
                 tp: int,
                 nproc_per_node: int,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda',
                 dtype: str = 'auto'):
        """initialize Executor."""
        super().__init__(model_path=model_path,
                         model_config=model_config,
                         cache_config=cache_config,
                         backend_config=backend_config,
                         tokenizer=tokenizer,
                         dp=dp,
                         tp=tp,
                         adapters=adapters,
                         device_type=device_type)

        self.world_size = tp * dp
        self.nproc_per_node = nproc_per_node
        device_ctx = DeviceContext(device_type)
        with get_device_manager().context(device_ctx):
            placement_group = init_ray_cluster(self.world_size)
        self.placement_group = placement_group
        self.master_addr = _get_master_addr()
        self.master_port = find_available_port()
        setup_master_addr(self.master_addr, self.master_port)

        # create workerwrapper actors
        worker_kwargs = dict(
            model_path=model_path,
            cache_config=cache_config,
            backend_config=backend_config,
            dp=dp,
            tp=tp,
            adapters=adapters,
            device_type=device_type,
            dtype=dtype,
            log_level=30,
            nproc_per_node=nproc_per_node,
        )
        self.workers = self._init_workers_ray(placement_group, worker_kwargs)
        self.dag = None

        # initialize process group
        ray.get([
            worker.init_process_group.remote(rank, self.master_addr, self.master_port)
            for rank, worker in enumerate(self.workers)
        ])

    def collective_rpc(self, method: str, args: Tuple[Any] = None, kwargs: Dict[str, Any] = None):
        """collective rpc."""
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        return ray.get([getattr(worker, method).remote(*args, **kwargs) for worker in self.workers])

    def build_model(self):
        """build model."""
        self.collective_rpc('build_model')

    def gather_free_mem(self):
        """gather available memory."""
        return self.collective_rpc('gather_free_mem')

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        self.collective_rpc('set_cache_config', (cache_config, ))

    def set_model_config(self, model_config: ModelConfig):
        """set all model config."""
        self.collective_rpc('set_model_config', (model_config, ))

    def build_graph_runner(self):
        """build graph runner."""
        self.collective_rpc('build_graph_runner')

    def build_cache_engine(self):
        """build cache engine."""
        self.collective_rpc('build_cache_engine')

    def get_input_processor(self):
        """build cache engine."""
        return ray.get(self.workers[0].get_input_processor.remote())

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        self.forward_event = forward_event
        self.collective_rpc('start')

    def stop(self):
        """stop engine loop."""
        self.collective_rpc('stop')

    def release(self):
        """release."""
        self.collective_rpc('release')
        for worker in self.workers:
            ray.kill(worker)
        if self.dag is not None:
            self.dag.teardown()
        ray.util.remove_placement_group(self.placement_group)

    async def forward_async(self, inputs):
        """start forward."""
        # we don't need return of forward async
        inputs = ray.put(inputs)
        [worker.forward_async.remote(inputs) for worker in self.workers]

    async def get_output_async(self):
        """get output async."""
        return await self.workers[0].get_output_async.remote()

    def _sort_workers(self, driver_ip: str, workers: List[RayWorkerWrapper]):
        """sort workers by ip."""
        worker_ips = ray.get([worker.get_node_ip.remote() for worker in workers])

        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        worker_ip_map = list(zip(workers, worker_ips))

        def sort_by_driver_then_worker_ip(item):
            """Sort the workers based on 3 properties:

            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item[1]
            return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_ip_map = sorted(worker_ip_map, key=sort_by_driver_then_worker_ip)
        workers = [item[0] for item in sorted_worker_ip_map]
        return workers

    def _init_workers_ray(self, placement_group: PlacementGroup, worker_kwargs: dict):
        """init worker ray."""
        device_str = get_device_str()
        bundle_indices = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if bundle.get(device_str, 0):
                bundle_indices.append(bundle_id)
        bundle_indices = bundle_indices[:self.world_size]

        workers = list()
        for _, bundle_id in enumerate(bundle_indices):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=1.0,
                scheduling_strategy=scheduling_strategy,
            )(RayWorkerWrapper).remote(**worker_kwargs)
            workers.append(worker)

        driver_ip = _get_master_addr()
        workers = self._sort_workers(driver_ip, workers)

        return workers
