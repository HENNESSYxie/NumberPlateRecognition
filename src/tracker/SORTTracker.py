import numpy as np
from tracker.sort import Sort
from utils.datamodels import Detection, Track


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
        result = []
        for i, tr in enumerate(tracks):
            tr = tr.astype(int)
            # result.append(Track(detection=detections[i], track_id=int(tr[4])))
            result.append(Track(detection=Detection(x1=tr[0], y1=tr[1], x2=tr[2], y2=tr[3], score=detections[i]),
                                track_id=tr[4]))
        return result

    def reset_ids(self):
        for t in self._tracker.trackers:
            t.id = 0
