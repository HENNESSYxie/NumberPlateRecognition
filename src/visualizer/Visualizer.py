import cv2

from typing import List
from utils.datamodels import Detection


class Visualizer:
    def visualize(self, image, detections: List[Detection]):
        for detection in detections:
            image = cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), (255, 0, 0), 2)
        return image
