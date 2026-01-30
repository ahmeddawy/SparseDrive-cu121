import numpy as np
import torch
from mmdet.registry import DATASETS

from projects.mmdet3d_plugin.compat import DataContainer as DC
from .nuscenes_3d_dataset import NuScenes3DDataset


@DATASETS.register_module()
class SparseDriveSequenceDataset(NuScenes3DDataset):
    """Dataset for loading sequences of frames for World Model training.

    Ensures temporal consistency by:
    1) Loading consecutive frames from the same scene
    2) Applying consistent augmentation across the sequence
    3) Rejecting invalid sequences (scene boundaries, missing data)
    """

    def __init__(
        self,
        queue_length=2,
        interval_2frames=False,
        max_skip_attempts=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.queue_length = queue_length
        self.interval_2frames = interval_2frames
        self.max_skip_attempts = max_skip_attempts

    def __getitem__(self, idx):
        if self.test_mode:
            return super().__getitem__(idx)

        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = None

        attempts = 0
        while attempts < self.max_skip_attempts:
            data = self.prepare_train_data(idx, aug_config=aug_config)
            if data is not None:
                return data
            idx = (idx + 1) % len(self.data_infos)
            attempts += 1

        raise RuntimeError(
            "Could not find a valid sequence after "
            f"{self.max_skip_attempts} attempts. "
            "Check data_infos ordering and scene continuity."
        )

    def prepare_train_data(self, index, aug_config=None):
        """Prepare a valid sequence of frames.

        Returns None if sequence is invalid.
        """
        data_queue = []

        if self.interval_2frames:
            frame_indices = list(range(index, index + self.queue_length))
        else:
            frame_indices = list(range(index - self.queue_length + 1, index + 1))

        if min(frame_indices) < 0 or max(frame_indices) >= len(self.data_infos):
            return None

        current_info = self.data_infos[index]
        current_scene = current_info.get("scene_token", None)
        if current_scene is None:
            return None

        for frame_idx in frame_indices:
            info = self.data_infos[frame_idx]
            if info.get("scene_token", None) != current_scene:
                return None

        if aug_config is None:
            aug_config = self.get_augmentation()

        for frame_idx in frame_indices:
            input_dict = self.get_data_info(frame_idx)
            if input_dict is None:
                return None
            input_dict["aug_config"] = aug_config
            example = self.pipeline(input_dict)
            data_queue.append(example)

        if len(data_queue) != self.queue_length:
            return None

        return self.union2one(data_queue)

    def union2one(self, queue):
        """Combine sequence of frames into batched format."""
        imgs_list = []
        metas_map = {}
        projection_mat_list = []

        for i, each in enumerate(queue):
            if "img" in each:
                imgs_list.append(each["img"].data)
            if "img_metas" in each:
                metas_map[i] = each["img_metas"].data
            if "projection_mat" in each:
                proj_mat = each["projection_mat"]
                if isinstance(proj_mat, DC):
                    proj_mat = proj_mat.data
                projection_mat_list.append(proj_mat)

        if len(imgs_list) == 0:
            return None

        res = queue[0]

        stacked_imgs = torch.stack(imgs_list)
        res["img"] = DC(stacked_imgs, cpu_only=False, stack=True)

        if len(projection_mat_list) >= 1:
            res["img_metas"].data["projection_mat_sequence"] = [
                torch.from_numpy(pm) if isinstance(pm, np.ndarray) else pm
                for pm in projection_mat_list
            ]
            current_proj = projection_mat_list[0]
            if isinstance(current_proj, np.ndarray):
                current_proj = torch.from_numpy(current_proj)
            res["projection_mat"] = DC(current_proj, cpu_only=False, stack=True)

        res["img_metas"].data["history"] = [
            metas_map[i] for i in range(len(queue))
        ]

        return res
