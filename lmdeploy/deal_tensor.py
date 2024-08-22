import torch
import torch_npu


def dump_tensor(x, name):
    import pickle
    with open(f'/data2/yaofengchen/workspaces/lmdeploy_InferExt/demo/{name}.pkl', 'wb') as f:
        if isinstance(x, torch.Tensor):
            pickle.dump(x.cpu(), f)
        else:
            pickle.dump(x, f)

def load_tensor(name):
    import pickle
    with open(f'/data2/yaofengchen/workspaces/mindie/mindie_demo/{name}.pkl', 'rb') as f:
        x = pickle.load(f)
    if isinstance(x, torch.Tensor):
        return x.npu()
    return x