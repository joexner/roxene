import abc

from roxene import PRECISION


class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self)-> PRECISION:
        pass