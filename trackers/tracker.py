from .deep_sort import DeepSort
from .sort import Sort
from .bytetrack import BYTETracker
import yaml
import numpy as np

supported = ["deepsort", "sort", "bytetrack"]


class ObjectTracker:
    def __init__(self, type):
        if type not in supported:
            raise TypeError(f"expected `type` in {supported}, but got {type}")

        with open(f"config/{type}.yaml", errors="ignore") as f:
            cfg = yaml.safe_load(f)
        if type == "deepsort":
            self.Tracker = DeepSort(**cfg)
            self.args = ["bboxes", "ori_img", 'cls']

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


def build_tracker(type):
    if type not in supported:
        raise TypeError(f"expected `type` in {supported}, but got {type}")

    with open(f"config/{type}.yaml", errors="ignore") as f:
        cfg = yaml.safe_load(f)
    if type == "deepsort":
        Tracker = DeepSort(**cfg)

    elif type == "bytetrack":
        Tracker = BYTETracker(**cfg)
    else:
        Tracker = Sort(**cfg)
    return Tracker
