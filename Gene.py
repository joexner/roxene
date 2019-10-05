import abc

from Organism import Organism


class Gene(abc.ABC):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass