import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..gene import Gene
from ..organism import Organism
from ..cell import Cell
from ..cells.neuron import Neuron


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
