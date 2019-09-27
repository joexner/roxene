import abc

class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self):
        pass