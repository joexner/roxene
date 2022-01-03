import abc
from collections import deque
from typing import Deque

from roxene import PRECISION, Cell


class Organism(object):

    def __init__(self):
        self.cells: Deque[Cell] = deque()

    def add(self, cell: Cell):
        self.cells.append(cell)


class Gene(abc.ABC):

    @abc.abstractmethod
    def execute(self, organism: Organism):
        pass