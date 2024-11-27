# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import Dict, Sequence

import torch
import zmq
from torch import multiprocessing as mp

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager

from ..model_inputs import ModelInputs
from .model_agent import AutoModelAgent, BaseModelAgent, SwapMap

zmq_ipc_url_prefix = 'ipc:///tmp/lmdeploy_pd_instance_'


def _start_pd_instance(
    instance_id: int,
    device_type: str,
    model_path: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    backend_config: BackendConfig,
    adapters: Dict[str, str] = None,
    trust_remote_code: bool = True,
):
    # init model_agent
    try:
        get_device_manager().set_context(DeviceContext(device_type))
        import torch

        from lmdeploy.pytorch.check_env import check_env_deeplink
        check_env_deeplink(device_type)
        torch.cuda.set_device(instance_id)
        model_agent = BaseModelAgent(
            model_path,
            model_config,
            cache_config,
            backend_config,
            adapters,
            trust_remote_code,
            instance_id,
        )
        # print(
        #     f'{instance_id}: model_agent cache: '
        #     f'{model_agent.cache_engine.gpu_cache[0][0].device}'
        # )
        method_map = {
            # "async_forward": model_agent.async_forward,
            'forward': model_agent.forward,
            'get_logits': model_agent.get_logits,
            'test': lambda *x: print(f'instance_id {instance_id} test: {x}')
        }
        ipc_url = zmq_ipc_url_prefix + str(instance_id)
        zmq_ctx = zmq.Context()
        rep_socket = zmq_ctx.socket(zmq.REP)
        rep_socket.bind(ipc_url)
        # hand shake confirm ready
        rep_socket.recv_pyobj()
        rep_socket.send_pyobj(model_agent.cache_config)

        while True:
            method_str, args = rep_socket.recv_pyobj()
            # print(f"instance {instance_id} "
            #       f"method_str: {method_str}, args: {args}")
            if method_str == 'stop':
                rep_socket.send_pyobj('stopped')
                zmq_ctx.destroy()
                break
            result = method_map[method_str](*args)
            rep_socket.send_pyobj(result)
        print(f'instance {instance_id} end')
    except Exception:
        import traceback
        traceback.print_exc()


class PDModelAgent(AutoModelAgent):

    def __init__(self,
                 num_instances: int,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 adapters: Dict[str, str] = None,
                 trust_remote_code: bool = True):
        super().__init__(model_config, cache_config)
        self.mp_ctx = mp.get_context('spawn')
        # self.mp_bar = self.mp_ctx.Barrier(num_instances + 1)
        self.mp_context = mp.spawn(
            _start_pd_instance,
            args=(
                'ascend',
                model_path,
                model_config,
                cache_config,
                backend_config,
                adapters,
                trust_remote_code,
            ),
            nprocs=num_instances,
            join=False,
            daemon=True,
        )
        self.stream = torch.cuda.Stream()

        self.zmq_ctx = zmq.Context()
        self.zmq_req_sockets: Sequence[zmq.SyncSocket] = []
        for idx in range(num_instances):
            zmq_req_socket = self.zmq_ctx.socket(zmq.REQ)
            zmq_req_socket.connect(zmq_ipc_url_prefix + str(idx))
            self.zmq_req_sockets.append(zmq_req_socket)
            # hand shake confirm ready
            zmq_req_socket.send_pyobj(None)
            instance_cache_config: CacheConfig = zmq_req_socket.recv_pyobj()
            num_cpu_blocks = instance_cache_config.num_cpu_blocks
            num_gpu_blocks = instance_cache_config.num_gpu_blocks
            if self.cache_config.num_cpu_blocks == 0 or \
                    num_cpu_blocks < self.cache_config.num_cpu_blocks:
                self.cache_config.num_cpu_blocks = num_cpu_blocks
            if self.cache_config.num_gpu_blocks == 0 or \
                    num_gpu_blocks < self.cache_config.num_gpu_blocks:
                self.cache_config.num_gpu_blocks = num_gpu_blocks

    def __del__(self):
        for idx, socket in enumerate(self.zmq_req_sockets):
            socket.send_pyobj(('stop', None))
            status = socket.recv_pyobj()
            if status == 'stopped':
                socket.unbind(zmq_ipc_url_prefix + str(idx))
            else:
                raise RuntimeError(
                    'sth wrong with '
                    f'{zmq_ipc_url_prefix + str(idx)} REP sides on exit')
        self.zmq_ctx.destroy()

    def get_logits(self,
                   hidden_states: torch.Tensor):  # , model_agent_id: int):
        model_agent_id = 1
        self.zmq_req_sockets[model_agent_id].send_pyobj(
            ('get_logits', (hidden_states, )))
        output = self.zmq_req_sockets[model_agent_id].recv_pyobj().to(0)
        return output

    async def async_forward(
            self,
            inputs: ModelInputs,  # model_agent_id: int,
            swap_in_map: SwapMap,
            swap_out_map: SwapMap):
        model_agent_id = 1
        # assert inputs.input_ids.device.type == 'cpu'
        self.zmq_req_sockets[model_agent_id].send_pyobj(
            ('forward', (inputs.to_device(model_agent_id), swap_in_map,
                         swap_out_map)))
        output = await asyncio.get_event_loop().run_in_executor(
            None, self.zmq_req_sockets[model_agent_id].recv_pyobj)
        return output
