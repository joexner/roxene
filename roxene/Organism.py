from roxene import Cell
from typing import Deque
from collections import deque

class Organism(object):

    def __init__(self):
        self.cells: Deque[Cell] = deque()

    def add(self, cell: Cell):
        self.cells.append(cell)
