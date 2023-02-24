from abc import abstractmethod, ABC
from pathlib import Path


class Decoder(ABC):
    @abstractmethod
    def __init__(self, path: Path):
        pass

    @abstractmethod
    def decode(self):
        pass
