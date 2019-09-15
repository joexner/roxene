from Cell import Cell


class Organism(object):

    def __init__(self):
        self.cells = []

    def add(self, cell: Cell):
        self.cells.append(cell)
