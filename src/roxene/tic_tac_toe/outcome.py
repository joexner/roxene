from enum import Enum, auto


class Outcome(Enum):
    WIN = auto()
    LOSE = auto()
    TIE = auto()
    TIMEOUT = auto()
    VALID_MOVE = auto()
    INVALID_MOVE = auto()
