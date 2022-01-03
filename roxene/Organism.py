from roxene import Cell
from typing import List

class Organism(object):

    def __init__(self):
        self.cells: List[Cell] = []

    def add(self, cell: Cell):
        self.cells.append(cell)
