import functools
import inspect
from collections.abc import Mapping, Sequence

import torch
from mmengine.dataset import default_collate
from mmengine.registry import MODEL_WRAPPERS
from mmdet.registry import MODELS


class DataContainer:
    """Lightweight replacement for mmcv.parallel.DataContainer."""

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=None):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    def __repr__(self):
        return (
            f"DataContainer(stack={self.stack}, cpu_only={self.cpu_only}, "
            f"pad_dims={self.pad_dims}, type={type(self.data)})"
        )


def _cast_tensor_type(data, dtype):
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    if isinstance(data, Mapping):
        return {k: _cast_tensor_type(v, dtype) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(_cast_tensor_type(v, dtype) for v in data)
    if isinstance(data, list):
        return [_cast_tensor_type(v, dtype) for v in data]
    return data


def auto_fp16(apply_to=None, out_fp32=False):
    """Best-effort replacement for mmcv.runner.auto_fp16."""
    apply_to = apply_to or []

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not getattr(args[0], "fp16_enabled", False):
                return func(*args, **kwargs)
            bound = sig.bind_partial(*args, **kwargs)
            for name in apply_to:
                if name in bound.arguments:
                    bound.arguments[name] = _cast_tensor_type(
                        bound.arguments[name], torch.float16
                    )
            output = func(*bound.args, **bound.kwargs)
            if out_fp32:
                output = _cast_tensor_type(output, torch.float32)
            return output

        return wrapper

    return decorator


def force_fp32(apply_to=None, out_fp16=False):
    """Best-effort replacement for mmcv.runner.force_fp32."""
    apply_to = apply_to or []

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not getattr(args[0], "fp16_enabled", False):
                return func(*args, **kwargs)
            bound = sig.bind_partial(*args, **kwargs)
            for name in apply_to:
                if name in bound.arguments:
                    bound.arguments[name] = _cast_tensor_type(
                        bound.arguments[name], torch.float32
                    )
            output = func(*bound.args, **bound.kwargs)
            if out_fp16:
                output = _cast_tensor_type(output, torch.float16)
            return output

        return wrapper

    return decorator


def wrap_fp16_model(model):
    """Compatibility shim; AMP is handled elsewhere in mmengine."""
    return model


def collate(batch, samples_per_gpu=1):
    """Collate function compatible with DataContainer."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"batch must be a sequence, got {type(batch)}")
    if len(batch) == 0:
        return batch

    first = batch[0]
    if isinstance(first, DataContainer):
        if first.cpu_only:
            return DataContainer([b.data for b in batch], cpu_only=True)
        if first.stack:
            return DataContainer(
                default_collate([b.data for b in batch]),
                stack=True,
                padding_value=first.padding_value,
                cpu_only=False,
            )
        return DataContainer([b.data for b in batch], stack=False)

    if isinstance(first, Mapping):
        return {k: collate([d[k] for d in batch], samples_per_gpu) for k in first}

    if isinstance(first, tuple):
        return [collate(samples, samples_per_gpu) for samples in zip(*batch)]

    if isinstance(first, list):
        return [collate(samples, samples_per_gpu) for samples in zip(*batch)]

    return default_collate(batch)


def scatter(inputs, devices, dim=0):
    """Minimal scatter compatible with DataContainer for single-device use."""
    if not isinstance(devices, (list, tuple)):
        devices = [devices]

    def _scatter(obj, device):
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            if obj.stack:
                return _scatter(obj.data, device)
            return [_scatter(x, device) for x in obj.data]
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, Mapping):
            return {k: _scatter(v, device) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(_scatter(v, device) for v in obj)
        if isinstance(obj, list):
            return [_scatter(v, device) for v in obj]
        return obj

    return [_scatter(inputs, device) for device in devices]


@MODEL_WRAPPERS.register_module()
class LegacyMMDataParallel(torch.nn.Module):
    """Compatibility wrapper for legacy data flow in single-device mode."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def train_step(self, data, optim_wrapper):
        device = next(self.module.parameters()).device
        data = scatter(data, [device])[0]
        losses = self.module(**data)
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data):
        device = next(self.module.parameters()).device
        data = scatter(data, [device])[0]
        return self.module(**data)

    def test_step(self, data):
        return self.val_step(data)


@MODEL_WRAPPERS.register_module()
class LegacyMMDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """Compatibility wrapper for legacy data flow in DDP mode."""

    def train_step(self, data, optim_wrapper):
        device = next(self.module.parameters()).device
        data = scatter(data, [device])[0]
        losses = self.module(**data)
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data):
        device = next(self.module.parameters()).device
        data = scatter(data, [device])[0]
        return self.module(**data)

    def test_step(self, data):
        return self.val_step(data)


MMDataParallel = LegacyMMDataParallel
MMDistributedDataParallel = LegacyMMDistributedDataParallel

ATTENTION = MODELS
POSITIONAL_ENCODING = MODELS
FEEDFORWARD_NETWORK = MODELS
NORM_LAYERS = MODELS
PLUGIN_LAYERS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_model(cfg):
    if cfg is None:
        return None
    return MODELS.build(cfg)


build_backbone = build_model
build_neck = build_model
build_head = build_model
build_loss = build_model


__all__ = [
    "DataContainer",
    "auto_fp16",
    "force_fp32",
    "wrap_fp16_model",
    "collate",
    "scatter",
    "LegacyMMDataParallel",
    "LegacyMMDistributedDataParallel",
    "MMDataParallel",
    "MMDistributedDataParallel",
    "ATTENTION",
    "POSITIONAL_ENCODING",
    "FEEDFORWARD_NETWORK",
    "NORM_LAYERS",
    "PLUGIN_LAYERS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
]
