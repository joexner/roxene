from roxene import Gene, Organism


class RotateCells(Gene):

    def __init__(self, direction):
        self.direction = direction
        pass

    def execute(self, organism: Organism):
        organism.cells.rotate(self.direction)
