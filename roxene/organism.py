import uuid
from collections import deque

from sqlalchemy.orm import Mapped, mapped_column
from typing import Deque

from persistence import EntityBase
from cells import Neuron, Cell, InputCell


class _SHMOrganism(EntityBase):
    __tablename__ = "organism"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)


    def __init__(self, input_names={}, output_names={}, genotype=None):
        self.id = uuid.uuid4()


class Organism(_SHMOrganism):

    def __init__(self, input_names={}, output_names={}, genotype=None):
        super().__init__(input_names, output_names, genotype)
        self.inputs: dict[str, InputCell] = dict((input_name, InputCell()) for input_name in input_names)
        self.outputs: dict[str, Neuron] = {}
        self.unused_output_names = list(output_names)
        self.cells: Deque[Cell] = deque(self.inputs.values())
        self.genotype = genotype
        if genotype:
            genotype.execute(self)

    def set_input(self, input_label, input_value):
        self.inputs[input_label].set_output(input_value)

    def get_output(self, output_label):
        return self.outputs[output_label].get_output()

    def update(self):
        for cell in self.cells:
            if isinstance(cell, Neuron):
                cell.update()

    def addNeuron(self, neuron: Neuron):
        self.cells.appendleft(neuron)
        if self.unused_output_names:
            new_output_name = self.unused_output_names.pop()
            self.outputs[new_output_name] = neuron

    def __str__(self):
        return f"O-{str(self.id)[-7:]}"

