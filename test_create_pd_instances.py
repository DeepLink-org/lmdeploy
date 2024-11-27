import multiprocessing

multiprocessing.Process

if __name__ == '__main__':
    from lmdeploy.pytorch.check_env import check_env_deeplink
    from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
    from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
    from lmdeploy.pytorch.engine.pd_agent import PDModelAgent
    check_env_deeplink('ascend')

    get_device_manager().set_context(DeviceContext('ascend'))

    model_path = '/pj_data/share/internlm_model/internlm2-chat-7b'
    trust_remote_code = True

    model_config = ModelConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, dtype='auto')
    model_config.custom_module_map = None

    cache_config = CacheConfig(max_batches=16,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               window_size=-1,
                               cache_max_entry_count=0.3,
                               max_prefill_token_num=4096,
                               enable_prefix_caching=False,
                               quant_policy=0)

    backend_config = BackendConfig(eager_mode=True, device_type='ascend')

    adapters = None

    model_agent = PDModelAgent(
        num_instances=2,
        model_path=model_path,
        model_config=model_config,
        cache_config=cache_config,
        backend_config=backend_config,
        adapters=adapters,
        trust_remote_code=trust_remote_code,
    )

    import torch
    model_agent.zmq_req_sockets[0].send_pyobj(
        ('test', ('data string', torch.tensor((2, 12), device='cuda:1'))))
    model_agent.zmq_req_sockets[1].send_pyobj(('test', ({
        1: 'world',
        2: '34'
    }, )))
    model_agent.zmq_req_sockets[1].recv_pyobj()
    model_agent.zmq_req_sockets[0].recv_pyobj()
    model_agent.zmq_req_sockets[0].send_pyobj(('test', ({
        3: 'ssss',
        '4': 'end'
    }, )))
    model_agent.zmq_req_sockets[0].recv_pyobj()

    del model_agent
    print('end')
