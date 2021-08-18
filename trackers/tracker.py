from .deep_sort.parser import get_config
from .deep_sort.deep_sort import DeepSort

from .sort import Sort
import os

supported = ['deepsort', 'sort']

class ObjectTracker:
    def __init__(self, name):
        if name not in supported:
            raise TypeError(f"expected `name` in {supported}, but got {name}")

        if name == 'deepsort':
            cfg = get_config()
            cfg.merge_from_file('trackers/deep_sort/deep_sort.yaml')
            self.Tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                max_dist=cfg.DEEPSORT.MAX_DIST, 
                                min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, 
                                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True)
        else:
            self.Tracker = Sort()

    def update(self, **kwargs):
        return self.Tracker.update(**kwargs)

