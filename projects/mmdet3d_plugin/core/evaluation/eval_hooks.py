import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.logging import print_log
from torch.nn.modules.batchnorm import _BatchNorm

from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list]
    )
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list]
    )
    return dynamic_milestones, dynamic_intervals


class CustomEvalHook(Hook):
    def __init__(
        self,
        dataloader,
        interval=1,
        by_epoch=True,
        start=None,
        dynamic_intervals=None,
        tmpdir=None,
        gpu_collect=False,
        eval_kwargs=None,
        broadcast_bn_buffer=True,
        save_best=None,
        rule=None,
    ):
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.start = 0 if start is None else start
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs or {}
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.save_best = save_best
        self.rule = rule

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            (
                self.dynamic_milestones,
                self.dynamic_intervals,
            ) = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            self.interval = self.dynamic_intervals[step - 1]

    def _should_evaluate(self, runner):
        if self.by_epoch:
            if (runner.epoch + 1) < self.start:
                return False
            return self.every_n_epochs(runner, self.interval, self.start)
        if (runner.iter + 1) < self.start:
            return False
        return self.every_n_train_iters(runner, self.interval, self.start)

    def before_train_epoch(self, runner):
        self._decide_interval(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._decide_interval(runner)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if not self._should_evaluate(runner):
            return
        self._do_evaluate(runner)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.by_epoch:
            return
        if not self._should_evaluate(runner):
            return
        self._do_evaluate(runner)

    def _get_eval_dataset(self):
        dataset = self.dataloader.dataset
        if hasattr(dataset, "evaluate"):
            return dataset
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "evaluate"):
            return dataset.dataset
        if hasattr(dataset, "datasets"):
            for ds in dataset.datasets:
                if hasattr(ds, "evaluate"):
                    return ds
        raise AttributeError("Dataset does not implement evaluate().")

    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if (
                    isinstance(module, _BatchNorm)
                    and module.track_running_stats
                ):
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
        )
        if runner.rank != 0:
            return

        dataset = self._get_eval_dataset()
        eval_kwargs = dict(self.eval_kwargs)
        eval_kwargs.setdefault("logger", runner.logger)
        eval_results = dataset.evaluate(results, **eval_kwargs)

        runner.message_hub.update_scalars(eval_results, resumed=False)
        print_log("\n", logger=runner.logger)
