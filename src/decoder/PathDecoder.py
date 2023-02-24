from pathlib import Path

from decoder.Decoder import  Decoder


class PathDecoder(Decoder):
    def __init__(self, path: Path):
        if not path.is_dir():
            raise NotADirectoryError(f"Path: {path} is not directory!")

        self._path = path

    def decode(self):
        for el in self._path.rglob("*"):
            if el.suffix == ".jpg" or el.suffix == ".png":
                yield el
