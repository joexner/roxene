import abc
from numpy import float16

class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self)-> float16:
        pass