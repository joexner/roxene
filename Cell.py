import abc
from tensorflow import float16

class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self)-> float16:
        pass