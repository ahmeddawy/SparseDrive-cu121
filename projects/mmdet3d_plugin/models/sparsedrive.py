from inspect import signature

import torch

from mmengine.registry import build_from_cfg
from mmdet.models import BaseDetector
from projects.mmdet3d_plugin.compat import (
    DETECTORS,
    PLUGIN_LAYERS,
    auto_fp16,
    build_backbone,
    build_head,
    build_neck,
    force_fp32,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    def _split_inputs(self, batch_inputs, batch_data_samples):
        if isinstance(batch_inputs, dict):
            data = dict(batch_inputs)
            img = data.pop("img")
            return img, data
        data = {}
        if isinstance(batch_data_samples, dict):
            data = dict(batch_data_samples)
        return batch_inputs, data

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        num_frames = 1
        if img.dim() == 6:  # (B, T, V, C, H, W)
            num_frames = img.shape[1]
            num_cams = img.shape[2]
            img = img.flatten(0, 2)
        elif img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            if num_frames > 1:
                feature_maps[i] = torch.reshape(
                    feat, (bs, num_frames, num_cams) + feat.shape[1:]
                )
            else:
                feature_maps[i] = torch.reshape(
                    feat, (bs, num_cams) + feat.shape[1:]
                )
        if return_depth and self.depth_branch is not None:
            if num_frames > 1:
                feat_curr = [f[:, 0] for f in feature_maps]
                depths = self.depth_branch(feat_curr, metas.get("focal"))
            else:
                depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def _forward(self, batch_inputs, batch_data_samples=None):
        img, data = self._split_inputs(batch_inputs, batch_data_samples)
        feature_maps = self.extract_feat(img, metas=data)
        if (
            self.use_deformable_func
            and isinstance(feature_maps, (list, tuple))
            and feature_maps[0].dim() == 5
        ):
            feature_maps = feature_maps_format(feature_maps)
        return self.head(feature_maps, data)

    def loss(self, batch_inputs, batch_data_samples):
        img, data = self._split_inputs(batch_inputs, batch_data_samples)
        return self.forward_train(img, **data)

    def predict(self, batch_inputs, batch_data_samples):
        img, data = self._split_inputs(batch_inputs, batch_data_samples)
        return self.simple_test(img, **data)

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        feature_maps_next = None
        projection_mat = data.get("projection_mat")
        if hasattr(projection_mat, "data"):
            projection_mat = projection_mat.data
        projection_mat_sequence = None

        # Save unformatted features for World Model
        feature_maps_raw = None
        feature_maps_next_raw = None

        if img.dim() == 6:
            num_frames = img.shape[1]
            feature_maps_curr = [f[:, 0] for f in feature_maps]
            feature_maps_next = []
            last_level = feature_maps[-1]
            for i in range(1, num_frames):
                feature_maps_next.append(last_level[:, i])

            # WM raw features: last level current + future
            feature_maps_raw = [feature_maps_curr[-1]]
            feature_maps_next_raw = [f.detach() for f in feature_maps_next]

            if self.use_deformable_func:
                feature_maps_curr = feature_maps_format(feature_maps_curr)
                feature_maps_next = feature_maps_format(feature_maps_next)

            feature_maps = feature_maps_curr

            # projection_mat_sequence from img_metas (cpu_only DataContainer)
            img_metas = data.get("img_metas", None)
            if isinstance(img_metas, list) and len(img_metas) > 0:
                first_meta = img_metas[0]
                if (
                    isinstance(first_meta, dict)
                    and "projection_mat_sequence" in first_meta
                ):
                    proj_seq_batch = []
                    for meta in img_metas:
                        if "projection_mat_sequence" in meta:
                            proj_seq_batch.append(meta["projection_mat_sequence"])
                    if len(proj_seq_batch) > 0:
                        num_frames_proj = len(proj_seq_batch[0])
                        projection_mat_sequence = []
                        for frame_idx in range(num_frames_proj):
                            frame_list = []
                            for proj_seq in proj_seq_batch:
                                frame_list.append(proj_seq[frame_idx])
                            projection_mat_sequence.append(
                                torch.stack(frame_list).to(feature_maps[0].device)
                            )
                        projection_mat = projection_mat_sequence[0]
        else:
            feature_maps_raw = (
                [f for f in feature_maps] if isinstance(feature_maps, list) else feature_maps
            )
            if (
                self.use_deformable_func
                and isinstance(feature_maps, (list, tuple))
                and feature_maps[0].dim() == 5
            ):
                feature_maps = feature_maps_format(feature_maps)

        model_outs = self.head(
            feature_maps,
            data,
            feature_maps_next=feature_maps_next,
            feature_maps_raw=feature_maps_raw,
            feature_maps_next_raw=feature_maps_next_raw,
            projection_mat=projection_mat,
            projection_mat_sequence=projection_mat_sequence,
        )
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        if img.dim() == 6 and img.shape[1] == 1:
            img = img.squeeze(1)
        feature_maps = self.extract_feat(img)

        if (
            self.use_deformable_func
            and DAF_VALID
            and isinstance(feature_maps, (list, tuple))
            and feature_maps[0].dim() == 5
        ):
            feature_maps = feature_maps_format(feature_maps)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
