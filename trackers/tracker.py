from .trackers.deep_sort import DeepSort
from .trackers.sort import Sort
from .trackers.bytetrack import BYTETracker
import yaml
import numpy as np
import os.path as osp

CONFIG_DIR = osp.join(
    osp.abspath(osp.join(osp.dirname(__file__), osp.pardir)), "config"
)

MODEL_DIR = osp.join(
    osp.abspath(osp.dirname(__file__)), "trackers/deep_sort/deep/checkpoint"
)

supported = ["deepsort", "sort", "bytetrack"]
deepsort_models = {128: "ckpt_128.t7", 512: "ckpt_512.t7"}


class ObjectTracker:
    def __init__(self, type, config=None):
        if type not in supported:
            raise TypeError(f"expected `type` in {supported}, but got {type}")

        config_file = config or osp.join(CONFIG_DIR, f"{type}.yaml")
        with open(config_file, errors="ignore") as f:
            cfg = yaml.safe_load(f)
        if type == "deepsort":
            model_path = osp.join(MODEL_DIR, deepsort_models[(cfg["feature_dim"])])
            self.Tracker = DeepSort(model_path=model_path, **cfg)
            self.args = ["bboxes", "ori_img", "cls"]

        elif type == "bytetrack":
            self.Tracker = BYTETracker(**cfg)
            self.args = ["bboxes", "scores"]
        else:
            self.Tracker = Sort(**cfg)
            self.args = ["bboxes", "cls"]

        self.type = type

    def update(self, **kwargs):
        outputs = self.Tracker.update(*[kwargs.get(a, None) for a in self.args])
        if self.type == "deepsort":
            tracks = outputs[0]
        elif self.type == "bytetrack":
            tracks = []
            for output in outputs:
                x1, y1, x2, y2 = output.tlbr
                tracks.append([x1, y1, x2, y2, output.track_id])
            if len(tracks):
                tracks = np.stack(tracks, axis=0)
        else:
            tracks = outputs
        return tracks


def build_tracker(type, config=None):
    if type not in supported:
        raise TypeError(f"expected `type` in {supported}, but got {type}")

    config_file = config or osp.join(CONFIG_DIR, f"{type}.yaml")
    with open(config_file, errors="ignore") as f:
        cfg = yaml.safe_load(f)
    if type == "deepsort":
        Tracker = DeepSort(**cfg)

    elif type == "bytetrack":
        Tracker = BYTETracker(**cfg)
    else:
        Tracker = Sort(**cfg)
    return Tracker
