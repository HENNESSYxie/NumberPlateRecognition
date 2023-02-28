import cv2
import numpy as np

from utils.OpenVINORuntime import OpenVINORuntime
from utils.datamodels import Detection


class SSDMobilenetV2OpenVINODetector(OpenVINORuntime):
    def _pre_processing(self, image):
        return cv2.resize(image, (self._width, self._height))[None, ...]

    def _post_processing(self, output, image):
        h, w = image.shape[:2]
        out = output[0]
        out = out[0][out[0][:, 2] > self._score_threshold, :]

        if len(out) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes=out[:, 3:].tolist(),
            scores=out[:, 2].tolist(),
            score_threshold=0.2,
            nms_threshold=0.5,
        )
        indices = np.array(indices, dtype=int).ravel()

        boxes = out[indices, 3:]
        boxes = np.hstack((boxes[:, :2] - boxes[:, 2:] // 2, boxes[:, 2:] + boxes[:, :2] // 2))
        boxes = ([w, h, w, h] * boxes).astype(int)
        boxes = boxes.clip(min=0, max=h if h == w else None)

        scores = out[indices, 2].tolist()
        labels = out[indices, 1].argmax(axis=0).tolist()

        detections = []
        for (ax1, ay1, ax2, ay2), score, label in zip(boxes, scores, [labels]):
            detections.append(
                Detection(x1=ax1, y1=ay1, x2=ax2, y2=ay2, score=score))
        return detections
