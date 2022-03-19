import abc
from collections import deque
from typing import Deque

from . import PRECISION


class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self) -> PRECISION:
        pass


class Organism(object):

    def __init__(self, **kwargs):
        if "cells" in kwargs:
            self.cells: Deque[Cell] = deque(kwargs["cells"])
        elif "genoytpe" in kwargs:
            self.cells: Deque[Cell] = deque()
            self.root_gene: Gene = kwargs["genoytpe"]
            self.root_gene.execute(self)
        else:
            self.cells: Deque[Cell] = deque()

    def setInput(self, input_label, input_value):
        pass

    def getOutput(self, output_label):
        pass


class Gene(abc.ABC):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass


class InputCell(Cell):

    def __init__(self, initial_value):
        self.value = initial_value

    def get_output(self) -> PRECISION:
        return self.value