import abc

from Organism import Organism


class Gene(abc.ABCMeta):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass