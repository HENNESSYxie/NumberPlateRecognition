import cv2
import numpy as np
from utils.OpenVINORuntime import OpenVINORuntime
from utils.datamodels import Detection


class YOLOv5PlateDetector(OpenVINORuntime):
    def _pre_processing(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (self._width, self._height), swapRB=True, crop=False)

    def _post_processing(self, output, image):
        original_height, original_width = image.shape[0], image.shape[1]
        out = output[0]  # `[x_center, y_center, width, height, score, classes...]`
        out = out[out[:, 4] > self._score_threshold, :]

        if len(out) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes=out[:, :4].tolist(),
            scores=out[:, 4].tolist(),
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
        )
        indices = np.array(indices, dtype=int).ravel()

        boxes = out[indices, :4]
        boxes = np.hstack((boxes[:, :2] - boxes[:, 2:] // 2, boxes[:, :2] + boxes[:, 2:] // 2))
        boxes = (boxes * [original_width / 640, original_height / 640, original_width / 640, original_height / 640]).astype(int)
        boxes = boxes.clip(min=0, max=original_height if original_height == original_width else None)

        scores = out[indices, 4].tolist()
        labels = out[indices, 5:].argmax(axis=1).tolist()

        detections = []

        for (ax1, ay1, ax2, ay2), score, label in zip(boxes, scores, labels):
            detections.append(
                Detection(x1=ax1, y1=ay1, x2=ax2, y2=ay2, score=score))
        return detections
