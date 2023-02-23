from dataclasses import dataclass


@dataclass
class Detection:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, score: float):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.score = score

