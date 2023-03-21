import numpy as np
from tracker.sort import Sort
from utils.datamodels import Detection


class SORTTracker:
    def __init__(self, min_hits: int, max_age: int):
        self._tracker = Sort(min_hits=min_hits, max_age=max_age)

    def update(self, detections):
        if len(detections) == 0:
            return []

        dets = []
        for detection in detections:
            dets.append(np.asarray([detection.x1, detection.y1,
                                    detection.x2, detection.y2, detection.score]))
        tracks = self._tracker.update(np.asarray(dets))
        return tracks

    def reset_ids(self):
        for t in self._tracker.trackers:
            t.id = 0
