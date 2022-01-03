import abc

from roxene import Organism


class Gene(abc.ABC):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass