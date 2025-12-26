from typing import List

from sqlalchemy.orm import Mapped, synonym

from ..gene import Gene
from ..genes.connect_neurons import ConnectNeurons
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene


class AddConnectNeurons(AddGene):
    __mapper_args__ = {"polymorphic_identity": "add_connect_neurons"}

    tx_cell_index: Mapped[int] = synonym("_i1")
    rx_port: Mapped[int] = synonym("_i2")

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01,
                 tx_cell_index: int = 0, rx_port: int = 0):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.tx_cell_index = tx_cell_index
        self.rx_port = rx_port

    def get_new_gene(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> Gene:
        return ConnectNeurons(self.tx_cell_index, self.rx_port)

