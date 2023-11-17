from collections import deque

import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict
from typing import Deque, Dict

from .cells import Cell, InputCell
from .neuron import Neuron
from .persistence import EntityBase


class Organism(EntityBase):
    __tablename__ = "organism"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)

    _inputs_map: Mapped[Dict[str, "_Organism_Input"]] = relationship(
        back_populates="organism",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("name"))

    input_cells: AssociationProxy[Dict[str, InputCell]] = association_proxy(
        target_collection="_inputs_map",
        attr="inputcell",
        creator=lambda name, inputcell: _Organism_Input(name=name, inputcell=inputcell))


    def __init__(self, input_names={}, output_names={}, genotype=None):
        self.id = uuid.uuid4()
        for name in input_names:
            self.input_cells[name] = InputCell()

        self.outputs: dict[str, Neuron] = {}
        self.unused_output_names = list(output_names)
        self.cells: Deque[Cell] = deque(self.input_cells.values())
        self.genotype = genotype
        if genotype:
            genotype.execute(self)

    def set_input(self, input_label, input_value):
        input_cell: InputCell = self.input_cells[input_label]
        input_cell.set_output(input_value)

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


class _Organism_Input(EntityBase):
    __tablename__ = "organism_input"

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"), primary_key=True)
    name: Mapped[str] = mapped_column(primary_key=True)
    inputcell_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("input_cell.id"))

    organism: Mapped[Organism] = relationship()
    inputcell: Mapped[InputCell] = relationship()

    def __init__(self, name, inputcell):
        self.name = name
        self.inputcell = inputcell
