# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import time
import os.path as osp

import torch
from mmengine.runner import Runner, set_random_seed

from mmdet.utils import compat_cfg

from projects.mmdet3d_plugin.compat import (
    LegacyMMDataParallel,
    LegacyMMDistributedDataParallel,
)
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomEvalHook
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset


def _build_optim_wrapper(cfg):
    optimizer_cfg = cfg.optimizer.copy()
    paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)
    if paramwise_cfg is not None:
        optim_wrapper = dict(
            type="OptimWrapperConstructor",
            optimizer=optimizer_cfg,
            paramwise_cfg=paramwise_cfg,
        )
    else:
        optim_wrapper = dict(optimizer=optimizer_cfg)
    optimizer_config = cfg.get("optimizer_config", {})
    if optimizer_config and optimizer_config.get("grad_clip") is not None:
        optim_wrapper["clip_grad"] = optimizer_config["grad_clip"]

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optim_wrapper["type"] = "AmpOptimWrapper"
        optim_wrapper["loss_scale"] = fp16_cfg.get("loss_scale", "dynamic")
    return optim_wrapper


def _build_param_scheduler(cfg, max_iters, by_epoch):
    lr_cfg = cfg.get("lr_config", None)
    if lr_cfg is None:
        return None
    if lr_cfg.get("policy") != "CosineAnnealing":
        raise ValueError(f'Unsupported lr_config policy {lr_cfg.get("policy")}')
    warmup_iters = lr_cfg.get("warmup_iters", 0)
    warmup_ratio = lr_cfg.get("warmup_ratio", 1.0)
    min_lr_ratio = lr_cfg.get("min_lr_ratio", 0.0)
    schedulers = []
    if warmup_iters > 0:
        schedulers.append(
            dict(
                type="LinearLR",
                start_factor=warmup_ratio,
                end_factor=1.0,
                by_epoch=by_epoch,
                begin=0,
                end=warmup_iters,
            )
        )
    schedulers.append(
        dict(
            type="CosineAnnealingLR",
            T_max=max_iters - warmup_iters,
            by_epoch=by_epoch,
            begin=warmup_iters,
            end=max_iters,
            eta_min_ratio=min_lr_ratio,
        )
    )
    return schedulers


def custom_train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    cfg = compat_cfg(cfg)

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    runner_type = cfg.runner["type"] if "runner" in cfg else "EpochBasedRunner"
    if runner_type == "IterBasedRunner":
        by_epoch = False
        train_cfg = dict(type="IterBasedTrainLoop", max_iters=cfg.runner.max_iters)
        max_iters = cfg.runner.max_iters
    else:
        by_epoch = True
        train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=cfg.runner.max_epochs)
        max_iters = cfg.runner.max_epochs

    samples_per_gpu = cfg.data.train_dataloader.get(
        "samples_per_gpu", cfg.data.get("samples_per_gpu", 1)
    )
    workers_per_gpu = cfg.data.train_dataloader.get(
        "workers_per_gpu", cfg.data.get("workers_per_gpu", 4)
    )
    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu,
            workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=dict(type="DistributedSampler"),
            runner_type=runner_type,
        )
        for ds in dataset
    ]
    train_dataloader = data_loaders[0]

    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = LegacyMMDataParallel(model)

    optim_wrapper = _build_optim_wrapper(cfg)
    param_scheduler = _build_param_scheduler(cfg, max_iters, by_epoch)

    default_hooks = dict()
    if cfg.get("checkpoint_config", None):
        default_hooks["checkpoint"] = dict(
            type="CheckpointHook",
            interval=cfg.checkpoint_config.get("interval", 1),
            by_epoch=by_epoch,
        )
    if cfg.get("log_config", None):
        default_hooks["logger"] = dict(
            type="LoggerHook",
            interval=cfg.log_config.get("interval", 10),
        )

    log_processor = dict(by_epoch=by_epoch)

    eval_hook = None
    if validate:
        val_cfg = cfg.data.val
        val_cfg.test_mode = True
        val_dataset = custom_build_dataset(val_cfg, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.data.val_dataloader.get("samples_per_gpu", 1),
            workers_per_gpu=workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {}).copy()
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        interval = eval_cfg.pop("interval", 1)
        dynamic_intervals = eval_cfg.pop("dynamic_intervals", None)
        start = eval_cfg.pop("start", None)
        eval_hook = CustomEvalHook(
            val_dataloader,
            interval=interval,
            by_epoch=by_epoch,
            start=start,
            dynamic_intervals=dynamic_intervals,
            eval_kwargs=eval_cfg,
        )

    env_cfg = dict(
        dist_cfg=cfg.get("dist_params", dict(backend="nccl")),
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    )

    launcher = cfg.get("launcher", "none")
    if distributed:
        model = LegacyMMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get("find_unused_parameters", False),
        )
    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        custom_hooks=[eval_hook] if eval_hook is not None else None,
        log_processor=log_processor,
        env_cfg=env_cfg,
        launcher=launcher if distributed else "none",
        load_from=cfg.get("load_from", None),
        resume=cfg.get("resume_from", None) is not None,
        randomness=dict(seed=cfg.seed, deterministic=cfg.get("deterministic", False)),
    )

    if cfg.get("resume_from", None):
        runner.resume(cfg.resume_from)
    runner.train()
