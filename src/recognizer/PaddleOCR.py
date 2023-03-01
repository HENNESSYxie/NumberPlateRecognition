import cv2

import utils.OCRprocessing as OCRprocessing
from utils.OpenVINORuntime import OpenVINORuntime
from utils.datamodels import Number


class PaddleOCR(OpenVINORuntime):
    def _pre_processing(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (self._width, self._height), swapRB=True, crop=False)

    def _post_processing(self, output, image) -> Number:
        out = OCRprocessing.build_post_process(OCRprocessing.postprocess_params)
        out = out(output)[0]
        if out[1] > self._score_threshold:
            return Number(number=out[0], score=out[1])
        return None
