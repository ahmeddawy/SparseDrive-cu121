import os

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking import mot as mot_module


def _patch_mot_empty_events():
    if getattr(mot_module, "_patched_empty_indices", False):
        return
    orig = mot_module.MOTAccumulatorCustom.new_event_dataframe_with_data

    def new_event_dataframe_with_data(indices, events):
        if indices is None:
            return mot_module.MOTAccumulatorCustom.new_event_dataframe()
        if isinstance(indices, dict):
            indices_list = list(indices.keys())
        else:
            try:
                indices_list = list(indices)
            except TypeError:
                indices_list = []
        if not indices_list:
            return mot_module.MOTAccumulatorCustom.new_event_dataframe()
        fixed = []
        for idx in indices_list:
            if isinstance(idx, (list, tuple)):
                if len(idx) == 0:
                    fixed.append((0, 0))
                elif len(idx) == 1:
                    fixed.append((idx[0], 0))
                else:
                    fixed.append((idx[0], idx[1]))
            else:
                fixed.append((idx, 0))
        return orig(fixed, events)

    mot_module.MOTAccumulatorCustom.new_event_dataframe_with_data = staticmethod(
        new_event_dataframe_with_data
    )
    mot_module._patched_empty_indices = True


class FilteredTrackingEval(TrackingEval):
    """Tracking eval that filters GT to predicted sample tokens."""

    def __init__(
        self,
        config,
        result_path,
        eval_set,
        output_dir,
        nusc_version,
        nusc_dataroot,
        verbose=True,
        render_classes=None,
    ):
        _patch_mot_empty_events()
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes

        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Initialize NuScenes object.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        # Load data.
        if verbose:
            print("Initializing nuScenes tracking evaluation")
        pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            TrackingBox,
            verbose=verbose,
        )
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        pred_tokens = set(pred_boxes.sample_tokens)
        filtered_gt = EvalBoxes()
        for token, boxes in gt_boxes.boxes.items():
            if token in pred_tokens:
                filtered_gt.add_boxes(token, boxes)

        kept = len(filtered_gt.boxes)
        dropped = len(gt_boxes.boxes) - kept
        gt_boxes = filtered_gt
        print(f"[info] Kept {kept} GT samples; dropped {dropped} with no predictions.")

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print("Filtering tracks")
        pred_boxes = filter_eval_boxes(
            nusc, pred_boxes, self.cfg.class_range, verbose=verbose
        )
        if verbose:
            print("Filtering ground truth tracks")
        gt_boxes = filter_eval_boxes(
            nusc, gt_boxes, self.cfg.class_range, verbose=verbose
        )

        self.sample_tokens = gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)
