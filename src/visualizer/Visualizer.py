import cv2

from typing import List
from utils.datamodels import Detection, Number, Track


class Visualizer:
    def visualize(self, image, track: Track, recognized_tracks: dict):
        image = cv2.rectangle(image, (track.detection.x1, track.detection.y1), (track.detection.x2, track.detection.y2),
                              (255, 0, 0), 2)
        image = cv2.putText(image, f"id:{track.track_id}", (track.detection.x1, track.detection.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 255, 150), 2, cv2.LINE_AA)
        if track.track_id in recognized_tracks.keys():
            image = cv2.putText(image, recognized_tracks[track.track_id].number, (track.detection.x2 - 100, track.detection.y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 255, 150), 2, cv2.LINE_AA)
        return image
