from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from typing import List

import uuid

from builtins import int
from enum import IntEnum, Enum, auto
from numpy import ndarray
from sqlalchemy import PickleType, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .persistence import EntityBase
from .cells import Cell, InputCell
from .neuron import Neuron
from .organism import Gene, Organism


class CNLayer(Enum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()


class CreateNeuron(Gene):
    __tablename__ = "create_neuron"
    __mapper_args__ = {"polymorphic_identity": "create_neuron"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)

    input: Mapped[ndarray] = mapped_column(PickleType)
    feedback: Mapped[ndarray] = mapped_column(PickleType)
    output: Mapped[ndarray] = mapped_column(PickleType)
    input_hidden: Mapped[ndarray] = mapped_column(PickleType)
    hidden_feedback: Mapped[ndarray] = mapped_column(PickleType)
    feedback_hidden: Mapped[ndarray] = mapped_column(PickleType)
    hidden_output: Mapped[ndarray] = mapped_column(PickleType)


    def __init__(self,
                 input: ndarray,
                 feedback: ndarray,
                 output: ndarray,
                 input_hidden: ndarray,
                 hidden_feedback: ndarray,
                 feedback_hidden: ndarray,
                 hidden_output: ndarray,
                 parent_gene: Gene = None):
        super().__init__(parent_gene)
        self.input = input
        self.feedback = feedback
        self.output = output
        self.input_hidden = input_hidden
        self.hidden_feedback = hidden_feedback
        self.feedback_hidden = feedback_hidden
        self.hidden_output = hidden_output


    def execute(self, organism: Organism):
        neuron = Neuron(
            input=self.input,
            feedback=self.feedback,
            output=self.output,
            input_hidden=self.input_hidden,
            hidden_feedback=self.hidden_feedback,
            feedback_hidden=self.feedback_hidden,
            hidden_output=self.hidden_output
        )
        organism.addNeuron(neuron)


class ConnectNeurons(Gene):
    __tablename__ = "connect_neurons"
    __mapper_args__ = {"polymorphic_identity": "connect_neurons"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    tx_cell_index: Mapped[int]
    rx_port: Mapped[int]

    def __init__(self, tx_cell_index, rx_input_port, parent_gene=None):
        super().__init__(parent_gene)
        self.tx_cell_index = tx_cell_index
        self.rx_port = rx_input_port

    def execute(self, organism: Organism):
        index: int = self.tx_cell_index % len(organism.cells)
        tx_cell: Cell = organism.cells[index]
        rx_cell: Neuron = organism.cells[0]
        rx_cell.add_input_connection(tx_cell, self.rx_port)


class RotateCells(Gene):
    __tablename__ = "rotate_cells"
    __mapper_args__ = {"polymorphic_identity": "rotate_cells"}

    class Direction(IntEnum):
        FORWARD = 1
        BACKWARD = -1

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    direction: Mapped[Direction] = mapped_column(Integer)

    def __init__(self, direction: Direction = Direction.BACKWARD, parent_gene=None):
        super().__init__(parent_gene)
        self.direction = direction

    def execute(self, organism: Organism):
        if self.direction is RotateCells.Direction.FORWARD:
            popped = organism.cells.pop()
            organism.cells.insert(0, popped)
        elif self.direction is RotateCells.Direction.BACKWARD:
            popped = organism.cells.pop(0)
            organism.cells.append(popped)


class CreateInputCell(Gene):
    __tablename__ = "create_input_cell"
    __mapper_args__ = {"polymorphic_identity": "create_input_cell"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    initial_value: Mapped[float] = mapped_column()

    def __init__(self, initial_value, parent_gene=None):
        super().__init__(parent_gene)
        self.initial_value = initial_value

    def execute(self, organism: Organism):
        input_cell = InputCell(self.initial_value)
        organism.cells.insert(0, input_cell)


class CompositeGene(Gene):
    __tablename__ = "composite_gene"
    __mapper_args__ = {"polymorphic_identity": "composite_gene"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    child_genes: Mapped[List[Gene]] = association_proxy(target_collection="_genes_list", attr="child")
    iterations: Mapped[int] = mapped_column()

    _genes_list: Mapped[List["_CompositeGene_Child"]] = relationship(
        back_populates="gene",
        cascade="all, delete-orphan",
        collection_class=ordering_list('ordinal'),
        lazy="select",
    )

    def __init__(self, child_genes: List[Gene], iterations: int = 1, parent_gene=None):
        super().__init__(parent_gene)
        self.child_genes = child_genes
        self.iterations = iterations

    def execute(self, organism: Organism):
        for n in range(self.iterations):
            for gene in self.child_genes:
                gene.execute(organism)


class _CompositeGene_Child(EntityBase):
    __tablename__ = "composite_gene_child"

    gene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("composite_gene.id"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)
    child_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"))

    gene: Mapped[CompositeGene] = relationship(foreign_keys=[gene_id])
    child: Mapped[Gene] = relationship(foreign_keys=[child_id])

    def __init__(self, child: Gene):
        self.child = child
