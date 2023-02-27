import cv2
import numpy as np

from utils.OpenVINORuntime import OpenVINORuntime


class EfficientNetV2sAligner(OpenVINORuntime):
    def _pre_processing(self, image):
        return np.expand_dims(
            cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0,
            axis=0)

    def _post_processing(self, output, image):
        (topleftX, topleftY, toprightX, toprightY, bottomrightX, bottomrightY, bottomleftX, bottomleftY) = tuple(output.reshape(8))
        (h, w) = image.shape[:2]

        topleftX = int(topleftX * w)
        topleftY = int(topleftY * h)
        toprightX = int(toprightX * w)
        toprightY = int(toprightY * h)
        bottomrightX = int(bottomrightX * w)
        bottomrightY = int(bottomrightY * h)
        bottomleftX = int(bottomleftX * w)
        bottomleftY = int(bottomleftY * h)

        rect = np.float32(
            ([topleftX, topleftY], [toprightX, toprightY], [bottomrightX, bottomrightY], [bottomleftX, bottomleftY]))
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (maxWidth, maxHeight))
        return warped
