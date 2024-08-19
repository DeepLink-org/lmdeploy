# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import re
from copy import copy
from typing import Any, Dict, Sequence

import torch
from addict import Addict

from lmdeploy.utils import get_logger

from ..devices import get_device_manager
from .module_map import DEVICE_SPECIAL_MODULE_MAP, MODULE_MAP

logger = get_logger('lmdeploy')


def _get_rewrite_qualname(origin_qualname: str, module_map: Dict[str,
                                                                 str]) -> str:
    """get rewrite module from origin module name.

    Args:
        origin_qualname (str): The origin qualname of the module.

    Returns:
        str: The rewrite qualname.
    """
    if origin_qualname in module_map:
        return module_map[origin_qualname]
    for key, value in module_map.items():
        if re.search(key, origin_qualname):
            return value
    return None


def _class_from_qualname(qualname: str) -> Any:
    """Import class with qualname.

    Args:
        qualname (str): Qualname of the class

    Returns:
        Any: class or builder of the class
    """
    last_dot = qualname.rfind('.')
    modname = qualname[:last_dot]
    clsname = qualname[last_dot + 1:]

    # get class at runtime
    mod = importlib.import_module(modname)
    assert mod is not None, f'failed to import module: {modname}'
    cls_type = getattr(mod, clsname)
    return cls_type


def _find_rewrite_module_qualname(model, module_map: Dict[str, str]):
    """find rewrite module."""
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__

    def _find_fullname():
        origin_qualname = f'{module_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_classname():
        origin_qualname = class_name
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_submodulename():
        # name with first module
        mod_name = module_name[module_name.rfind('.') + 1:]
        origin_qualname = f'{mod_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    rewrite_qualname = _find_fullname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_classname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_submodulename()

    origin_qualname = f'{module_name}.{class_name}'
    if rewrite_qualname is not None:
        logger.debug('Find rewrite of module\n'
                     f'{origin_qualname} <=> {rewrite_qualname}')
    return rewrite_qualname


def _update_module_type(model: Any, cls_type: type, custom_attrs: dict = None):
    """Update class type of model."""
    # directly return origin model is not cool
    # origin model would be registered as a submodule
    old_type = type(model)

    @property
    def get_origin_mod(self):
        origin_mod = copy(self)
        origin_mod.__class__ = old_type
        return origin_mod

    attrs = dict(cls_type.__dict__)
    custom_attrs = custom_attrs or dict()
    custom_attrs['origin_mod'] = get_origin_mod
    attrs.update(custom_attrs)
    new_type = type(cls_type.__name__, (type(model), ), attrs)
    model = copy(model)
    model.__class__ = new_type

    return model


def _patch(model: torch.nn.Module,
           context: Addict,
           module_map: Dict[str, str] = None) -> torch.nn.Module:
    """patch the model with rewrite module.

    Args:
        model (Module): model to be patched.
        context (Addict): The environment info to patched in model

    Returns:
        Module: The patched model
    """

    if module_map is None:
        module_map = MODULE_MAP

    def _recursive_children(context, named_children):
        """recursive children."""
        for name, child in named_children:
            patched_child = _patch(child, context, module_map=module_map)
            if patched_child != child:
                model.register_module(name, patched_child)

    _recursive_children(context, model.named_children())
    rewrite_qualname = _find_rewrite_module_qualname(model,
                                                     module_map=module_map)

    if rewrite_qualname is not None:
        cls_type = _class_from_qualname(rewrite_qualname)
        model = _update_module_type(model, cls_type, dict(context=context))

    return model


def _update_model(model: torch.nn.Module):
    """Update model after patch and load.

    Args:
        model (Module): The model to be updated.
    """
    # recursive over children
    for _, child in model.named_children():
        _update_model(child)

    torch.cuda.empty_cache()
    # trans weights
    if hasattr(model, "weight"):
        if isinstance(model.weight, torch.nn.parameter.Parameter):
            model.weight.data = model.weight.data.contiguous()
        elif isinstance(model.weight, torch.Tensor):
            model.weight = model.weight.contiguous()
        else:
            raise ValueError(f"Unsupported weight type: {type(model.weight)}.")
    # if hasattr(model, "q_proj"):
    #     model.q_proj.weight.data = model.q_proj.weight.data.t().contiguous()
    # if hasattr(model, "k_proj"):
    #     model.k_proj.weight.data = model.k_proj.weight.data.t().contiguous()
    # if hasattr(model, "v_proj"):
    #     model.v_proj.weight.data = model.v_proj.weight.data.t().contiguous()
    # if hasattr(model, 'o_proj'):
    #     model.o_proj.weight.data = model.o_proj.weight.data.t().contiguous()
    # if hasattr(model, "q_proj") and hasattr(model, "k_proj") and hasattr(model, 'v_proj'):
    #     qkv = torch.cat((model.q_proj.weight, model.k_proj.weight, model.v_proj.weight), dim=-1)
    #     # del model.q_proj, model.k_proj, model.v_proj
    #     head_dim = model.head_dim
    #     kv_groups = model.num_key_value_groups
    #     size = qkv.shape[0]
    #     qkv = qkv.reshape(size, -1, kv_groups + 2, head_dim)
    #     wq, wk, wv = qkv.split([kv_groups, 1, 1], dim=-2)
    #     del qkv
    #     wq = wq.reshape(size, -1)
    #     wk = wk.reshape(size, -1)
    #     wv = wv.reshape(size, -1)
    #     model.qkv = torch.cat([wq, wk, wv], dim=-1)
    #     del wq, wk, wv

    torch.cuda.empty_cache()
    # if hasattr(model, 'gate_proj') and not hasattr(model, "dense_h_to_4h"):
    #     # imporct pdb; pdb.set_trace()
    #     model.gate_proj.weight.data = model.gate_proj.weight.data.t().contiguous()
    # if hasattr(model, 'up_proj'):
    #     model.up_proj.weight.data = model.up_proj.weight.data.t().contiguous()
    # if hasattr(model, 'down_proj'):
    #     model.down_proj.weight.data = model.down_proj.weight.data.t().contiguous()
    # if hasattr(model, 'gate_proj') and hasattr(model, 'up_proj'):
    #     # import pdb; pdb.set_trace()
    #     model.trans_wgate_up = torch.cat((model.gate_proj.weight, model.up_proj.weight), dim=-1)
    #     del model.gate_proj
    #     del model.up_proj

    if hasattr(model, 'wqkv'):
        wqkv = model.wqkv.weight.data.t().contiguous()
        del model.wqkv
        head_dim = model.head_dim
        kv_groups = model.num_key_value_groups
        size = wqkv.shape[0]
        wqkv = wqkv.reshape(size, -1, kv_groups + 2, head_dim)
        wq, wk, wv = wqkv.split([kv_groups, 1, 1], dim=-2)
        wq = wq.reshape(size, -1)
        wk = wk.reshape(size, -1)
        wv = wv.reshape(size, -1)
        model.wqkv = torch.cat([wq, wk, wv], dim=-1)
        del wq, wk, wv

    if hasattr(model, 'wo'):
        model.wo.weight.data= model.wo.weight.data.t().contiguous()
    if hasattr(model, 'w1'):
        model.w1.weight.data = model.w1.weight.data.t().contiguous()
    if hasattr(model, 'w2'):
        model.w2.weight.data = model.w2.weight.data.t().contiguous()
    if hasattr(model, 'w3'):
        model.w3.weight.data = model.w3.weight.data.t().contiguous()
    if hasattr(model, 'w1') and hasattr(model, 'w3'):
        model.trans_w13 = torch.cat((model.w1.weight, model.w3.weight), dim=-1)
        del model.w1, model.w3
    if hasattr(model, 'language_expert_dense'):
        # import pdb; pdb.set_trace()
        model.language_expert_dense.weight.data = model.language_expert_dense.weight.data.t().contiguous()
    if hasattr(model, 'language_expert_query_key_value'):
        # import pdb; pdb.set_trace()
        model.language_expert_query_key_value.weight.data = model.language_expert_query_key_value.weight.data.t().contiguous()
    if hasattr(model, 'vision_expert_dense'):
        # import pdb; pdb.set_trace()
        model.vision_expert_dense.weight.data = model.vision_expert_dense.weight.data.t().contiguous()
    if hasattr(model, 'vision_expert_query_key_value'):
        # import pdb; pdb.set_trace()
        model.vision_expert_query_key_value.weight.data = model.vision_expert_query_key_value.weight.data.t().contiguous()
    
    # cogvlm visual
    if hasattr(model, "query_key_value"):
        model.query_key_value.weight.data = model.query_key_value.weight.data.t().contiguous()
    if hasattr(model, "dense"):
        model.dense.weight.data = model.dense.weight.data.t().contiguous()
    if hasattr(model, "fc1"):
        model.fc1.weight.data = model.fc1.weight.data.t().contiguous()
    if hasattr(model, "fc2"):
        model.fc2.weight.data = model.fc2.weight.data.t().contiguous()
    if hasattr(model, "linear_proj") and hasattr(model.linear_proj, "weight"):
        # import pdb; pdb.set_trace()
        model.linear_proj.weight.data = model.linear_proj.weight.data.t().contiguous()
    if hasattr(model, "gate_proj") and hasattr(model, "dense_h_to_4h"):
        # gate_proj.weight.data has already transposed.
        model.gate_proj.weight.data = model.gate_proj.weight.data.t().contiguous()
        model.dense_h_to_4h.weight.data = model.dense_h_to_4h.weight.data.t().contiguous()
        model.gate_dense_weight = torch.cat((model.gate_proj.weight, model.dense_h_to_4h.weight), dim=-1)
    if hasattr(model, "dense_4h_to_h"):
        model.dense_4h_to_h.weight.data = model.dense_4h_to_h.weight.data.t().contiguous()

    if hasattr(model, '_update_model_fn'):
        model._update_model_fn()


def update_model(model: torch.nn.Module):
    """update model."""
    return _update_model(model)


def _dist_model(model: torch.nn.Module, rank: int = 0):
    """distribute model parameters."""

    def _register_hooks():
        """register hooks."""
        if hasattr(model, '_distribute_input_fn'):
            input_fn = model._distribute_input_fn
            model.register_forward_pre_hook(
                lambda _, inputs, inputs_dict: input_fn(inputs, inputs_dict),
                with_kwargs=True,
            )

        if hasattr(model, '_distribute_output_fn'):
            output_fn = model._distribute_output_fn
            model.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs))

    for name, child in model.named_children():
        if rank == 0:
            logger.debug(f'Distribute module: <{name}>')
        new_child = _dist_model(child, rank)
        if new_child != child:
            model.register_module(name, child)

    _register_hooks()

    return model


class PatchedForward:
    """patched forward."""

    def __init__(self, model, context, extra_args):
        self._model = model
        self._patch_context: Dict = context
        self._extra_args: list = extra_args

    def __call__(self, *args, **kwargs):
        for arg_name in self._extra_args:
            extra_arg = kwargs.pop(arg_name, None)
            self._patch_context[arg_name] = extra_arg

        output = self._model(*args, **kwargs)

        self._patch_context.clear()

        return output


@torch.inference_mode()
def patch(
    model: torch.nn.Module,
    extra_args: Sequence[str] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Patch the model with rewrite modules.

    Extra arguments will be patched in forward of model, weights on each rank
    will be partitioned.

    Args:
        model (Module): Model to be patched.
        extra_args (Sequence[str]): Extra arguments of model forward.
        rank (int): Distribution rank.
        world_size (int): Distribution world size.

    Returns:
        Module: The patched model.
    """
    if rank == 0:
        logger.info('Patching model.')

    if extra_args is None:
        extra_args = []

    _patch_context = Addict()

    module_map = MODULE_MAP.copy()
    device_type = get_device_manager().current_context().device_type
    if device_type != 'cuda':
        device_map = DEVICE_SPECIAL_MODULE_MAP.get(device_type, dict())
        module_map.update(device_map)

    model = _patch(model, _patch_context, module_map=module_map)

    if world_size > 1:
        model = _dist_model(model, rank)

    patched_forward = PatchedForward(model,
                                     _patch_context,
                                     extra_args=extra_args)
    model.patched_forward = patched_forward

    return model
