import abc
from collections import deque
from typing import Deque

from roxene import PRECISION


class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self)-> PRECISION:
        pass


class Organism(object):

    def __init__(self):
        self.cells: Deque[Cell] = deque()

    def add(self, cell: Cell):
        self.cells.append(cell)


class Gene(abc.ABC):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass


class InputCell(Cell):

    def __init__(self, initial_value):
        self.value = initial_value

    def get_output(self):
        return self.value