import os

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval


class FilteredDetectionEval(DetectionEval):
    """Detection eval that filters GT to predicted sample tokens."""

    def __init__(
        self,
        nusc,
        config,
        result_path,
        eval_set,
        output_dir=None,
        verbose=True,
    ):
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose,
        )
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        pred_tokens = set(self.pred_boxes.sample_tokens)
        filtered_gt = EvalBoxes()
        for token, boxes in self.gt_boxes.boxes.items():
            if token in pred_tokens:
                filtered_gt.add_boxes(token, boxes)

        kept = len(filtered_gt.boxes)
        dropped = len(self.gt_boxes.boxes) - kept
        self.gt_boxes = filtered_gt
        print(f"[info] Kept {kept} GT samples; dropped {dropped} with no predictions.")

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print("Filtering predictions")
        self.pred_boxes = filter_eval_boxes(
            nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose
        )
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = filter_eval_boxes(
            nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose
        )

        self.sample_tokens = self.gt_boxes.sample_tokens
