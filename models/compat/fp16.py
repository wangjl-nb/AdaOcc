import functools
import inspect
from typing import Iterable

import torch

__all__ = [
    'auto_fp16',
    'force_fp32',
    'cast_tensor_type',
    'multi_apply',
]


def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    if isinstance(inputs, tuple):
        return tuple(cast_tensor_type(x, src_type, dst_type) for x in inputs)
    if isinstance(inputs, list):
        return [cast_tensor_type(x, src_type, dst_type) for x in inputs]
    if isinstance(inputs, dict):
        return {k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()}
    return inputs


def _to_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def auto_fp16(apply_to=None, out_fp32=False):
    apply_to = _to_tuple(apply_to)

    def decorator(func):
        arg_spec = inspect.getfullargspec(func)
        arg_names = arg_spec.args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not getattr(args[0], 'fp16_enabled', False):
                return func(*args, **kwargs)

            args = list(args)
            for name in apply_to:
                if name in kwargs:
                    kwargs[name] = cast_tensor_type(kwargs[name], torch.float32, torch.float16)
                elif name in arg_names:
                    idx = arg_names.index(name)
                    if idx < len(args):
                        args[idx] = cast_tensor_type(args[idx], torch.float32, torch.float16)
            output = func(*args, **kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.float16, torch.float32)
            return output

        return wrapper

    return decorator


def force_fp32(apply_to=None, out_fp32=False):
    apply_to = _to_tuple(apply_to)

    def decorator(func):
        arg_spec = inspect.getfullargspec(func)
        arg_names = arg_spec.args

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            for name in apply_to:
                if name in kwargs:
                    kwargs[name] = cast_tensor_type(kwargs[name], torch.float16, torch.float32)
                elif name in arg_names:
                    idx = arg_names.index(name)
                    if idx < len(args):
                        args[idx] = cast_tensor_type(args[idx], torch.float16, torch.float32)
            output = func(*args, **kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.float16, torch.float32)
            return output

        return wrapper

    return decorator


def multi_apply(func, *args, **kwargs):
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
