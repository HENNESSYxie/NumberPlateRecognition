from utils.datamodels import Track


class PassHeuristicWithLine:
    def __init__(self, line: float):
        self._line = line
        self.recognized_tracks = {}

    def run(self, image, track: Track):
        h, w = image.shape[:2]
        if track.detection.x1 > int(self._line * w) and track.track_id not in self.recognized_tracks.keys():
            return True

    def clear_dict(self):
        self.recognized_tracks.clear()
