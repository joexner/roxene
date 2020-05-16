from Cell import Cell


class InputCell(Cell):

    def __init__(self, initial_value):
        self.value = initial_value

    def get_output(self):
        return self.value