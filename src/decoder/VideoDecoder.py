import cv2
from pathlib import Path

from decoder.Decoder import Decoder


class VideoDecoder(Decoder):
    def __init__(self, path: Path):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Video by path: {path} not found!')

        self._path = path

    def decode(self):
        capture = cv2.VideoCapture(self._path)
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield frame
