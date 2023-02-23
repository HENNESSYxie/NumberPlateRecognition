from pathlib import Path


class VideoDecoder:
    def __init__(self, path: Path):
        if not path.is_file():
            print(f"Video by path {path} not found")

        self._path = path

    def decode(self):
        pass
