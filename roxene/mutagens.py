import abc
from abc import ABC

from roxene import CreateNeuron, Gene


class Mutagen(ABC):

    def __init__(self, severity):
        self.severity = severity

    def mutate(self, gene: Gene) -> Gene:
        return gene

    def should_mutate(self, gene) -> bool:
        return False

class NeuronInitialValueMutagen(Mutagen):

    def mutate(self, gene: CreateNeuron):
        if self.should_mutate(gene):
            gene.input_initial_value
        else:
            return gene


    def should_mutate(self, gene):
        random.
