from __future__ import annotations

import abc
import uuid
from typing import Dict, List, Optional

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict

from .cells import Cell, InputCell
from .neuron import Neuron
from .persistence import EntityBase


# Racka-fracka cyclic imports...
class Gene(EntityBase):
    __tablename__ = "gene"
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    parent_gene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("gene.id"))
    parent_gene: relationship(Gene, remote_side=[id])
    type: Mapped[str]

    __mapper_args__ = {
        "polymorphic_identity": "gene",
        "polymorphic_on": "type",
    }

    def __init__(self, parent_gene=None):
        self.parent_gene = parent_gene
        self.id = uuid.uuid4()

    @abc.abstractmethod
    def execute(self, organism: 'Organism'):
        pass

    def __str__(self):
        return f"G-{str(self.id)[-7:]}"


class _Organism_Input(EntityBase):
    __tablename__ = "organism_input"

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"), primary_key=True)
    name: Mapped[str] = mapped_column(primary_key=True)
    inputcell_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("input_cell.id"))

    organism: Mapped["Organism"] = relationship()
    inputcell: Mapped[InputCell] = relationship(lazy="joined")

    def __init__(self, name, inputcell):
        self.name = name
        self.inputcell = inputcell


class _Organism_Output(EntityBase):
    __tablename__ = "organism_output"

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"), primary_key=True)
    name: Mapped[str] = mapped_column(primary_key=True)
    neuron_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("neuron.id"))

    organism: Mapped["Organism"] = relationship()
    neuron: Mapped[Neuron] = relationship(lazy="joined")

    def __init__(self, name, neuron):
        self.name = name
        self.neuron = neuron


class _Organism_Cell(EntityBase):
    __tablename__ = "organism_cell"

    organism_id = mapped_column(ForeignKey("organism.id"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)
    cell_id = mapped_column(ForeignKey("cell.id"), primary_key=True)

    organism: Mapped["Organism"] = relationship()
    cell: Mapped[Cell] = relationship(cascade="all", lazy="joined")

    def __init__(self, cell):
        self.cell = cell


class _Organism_Unused_Output_Name(EntityBase):
    __tablename__ = "organism_unused_output_name"

    organism_id = mapped_column(ForeignKey("organism.id"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)

    name = mapped_column(String())
    organism: Mapped["Organism"] = relationship()

    def __init__(self, name):
        self.name = name


class Organism(EntityBase):
    __tablename__ = "organism"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)

    cells: Mapped[List[Cell]] = association_proxy(target_collection="_cells_list", attr="cell", creator=_Organism_Cell)
    inputs: Mapped[Dict[str, InputCell]] = association_proxy(target_collection="_inputs_map", attr="inputcell",
                                                             creator=_Organism_Input)
    outputs: Mapped[Dict[str, Neuron]] = association_proxy(target_collection="_outputs_map", attr="neuron",
                                                           creator=_Organism_Output)

    genotype: Mapped[Optional[Gene]] = relationship()
    unused_output_names: Mapped[List[str]] = association_proxy(target_collection="_unused_output_names_list",
                                                               attr="name", creator=_Organism_Unused_Output_Name)

    genotype_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("gene.id"))

    _inputs_map: Mapped[Dict[str, _Organism_Input]] = relationship(
        back_populates="organism",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("name"),
        lazy="joined",
    )

    _outputs_map: Mapped[Dict[str, _Organism_Output]] = relationship(
        back_populates="organism",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("name"),
        lazy="joined",
    )

    _cells_list: Mapped[List[_Organism_Cell]] = relationship(
        back_populates="organism",
        order_by="_Organism_Cell.ordinal",
        collection_class=ordering_list('ordinal'),
        cascade="all, delete-orphan",
        lazy="joined",
    )

    _unused_output_names_list: Mapped[List[_Organism_Unused_Output_Name]] = relationship(
        back_populates="organism",
        order_by="_Organism_Unused_Output_Name.ordinal",
        collection_class=ordering_list('ordinal'),
        cascade="all, delete-orphan",
        lazy="joined",
    )

    def __init__(self, input_names={}, output_names={}, genotype: Gene = None):
        self.id = uuid.uuid4()
        for name in input_names:
            self.inputs[name] = InputCell()
        self.unused_output_names.extend(output_names)
        self.cells.extend(self.inputs.values())
        self.genotype = genotype
        if genotype:
            genotype.execute(self)

    def set_input(self, input_label, input_value):
        input_cell: InputCell = self.inputs[input_label]
        input_cell.set_output(input_value)

    # TODO: Make output a @property, and maybe input and cells
    def get_output(self, output_label) -> float:
        return float(self.outputs[output_label].get_output())

    def update(self):
        for cell in self.cells:
            if isinstance(cell, Neuron):
                cell.update()

    def addNeuron(self, neuron: Neuron):
        self.cells.insert(0, neuron)
        if self.unused_output_names:
            new_output_name = self.unused_output_names.pop()
            self.outputs[new_output_name] = neuron

    def __str__(self):
        return f"O-{str(self.id)[-7:]}"

