import pytest
from utils.config_utils import get_communicator_list, get_turbomind_model_list, get_workerid
from utils.run_restful_chat import run_all_step, start_restful_api, stop_restful_api

DEFAULT_PORT = 23333


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    param = request.param
    model = param['model']
    model_path = config.get('model_path') + '/' + model

    pid, startRes = start_restful_api(config, param, model, model_path, 'turbomind', worker_id)
    yield
    stop_restful_api(pid, startRes, param)


def getModelList(tp_num):
    model_list = []
    for communicator in get_communicator_list():
        model_list += [{
            'model': item,
            'cuda_prefix': None,
            'tp_num': tp_num,
            'extra': f'--communicator {communicator}'
        } for item in get_turbomind_model_list(tp_num)]
    return model_list


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=1), indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=2), indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getModelList(tp_num=4), indirect=True)
def test_restful_chat_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


def getKvintModelList(tp_num, quant_policy):
    model_list = []
    for communicator in get_communicator_list():
        model_list += [{
            'model': item,
            'cuda_prefix': None,
            'tp_num': tp_num,
            'extra': f'--quant-policy {quant_policy} --communicator {communicator}'
        } for item in get_turbomind_model_list(tp_num, quant_policy=quant_policy)]
    return model_list


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=4), indirect=True)
def test_restful_chat_kvint4_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=1, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=2, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp2(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', getKvintModelList(tp_num=4, quant_policy=8), indirect=True)
def test_restful_chat_kvint8_tp4(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', [
    {
        'model': 'internlm/internlm2_5-20b-chat',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
    {
        'model': 'internlm/internlm2_5-20b-chat-inner-4bits',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
    {
        'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'cuda_prefix': 'CUDA_VISIBLE_DEVICES=5,6',
        'tp_num': 2
    },
],
                         indirect=True)
def test_restful_chat_pr(config, common_case_config):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, {key: value for key, value in case_config.items() if key == 'memory_test'})


@pytest.mark.order(7)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.restful_api
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'cuda_prefix': None,
    'tp_num': 1,
    'modelscope': True
}],
                         indirect=True)
def test_modelscope_restful_chat_tp1(config, common_case_config, worker_id):
    if get_workerid(worker_id) is None:
        run_all_step(config, common_case_config)
    else:
        run_all_step(config, common_case_config, worker_id=worker_id, port=DEFAULT_PORT + get_workerid(worker_id))
