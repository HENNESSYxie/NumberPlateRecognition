import cv2

from typing import List
from utils.datamodels import Detection, Number


class Visualizer:
    def visualize(self, image, detection: Detection, number: Number):
        image = cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), (255, 0, 0), 2)
        if number is not None:
            image = cv2.putText(image, number.number, (detection.x2 - 40, detection.y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (235, 255, 150), 2, cv2.LINE_AA)
        return image
