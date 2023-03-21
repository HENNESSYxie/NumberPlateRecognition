import cv2

from typing import List
from utils.datamodels import Detection, Number, Track


class Visualizer:
    def visualize(self, image, track: Track, number: Number):
        image = cv2.rectangle(image, (track.detection.x1, track.detection.y1), (track.detection.x2, track.detection.y2),
                              (255, 0, 0), 2)
        image = cv2.putText(image, str(track.track_id), (track.detection.x1, track.detection.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 255, 150), 2, cv2.LINE_AA)
        if number is not None:
            image = cv2.putText(image, number.number, (track.detection.x2 - 40, track.detection.y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 255, 150), 2, cv2.LINE_AA)
        return image
