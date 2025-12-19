import uuid
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..gene import Gene
from ..genes.connect_neurons import ConnectNeurons
from ..genes.composite_gene import CompositeGene
from .insert_gene_to_composite_mutagen import InsertGeneToCompositeMutagen
from ..util import get_rng


class AddConnectionMutagen(InsertGeneToCompositeMutagen):
    """
    Adds a new ConnectNeurons gene to a CompositeGene.
    This creates new connections between cells in the organism.
    """
    __tablename__ = "add_connection_mutagen"
    __mapper_args__ = {"polymorphic_identity": "add_connection_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("insert_gene_to_composite_mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def get_genes_to_insert(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> List[Gene]:
        """Create a new ConnectNeurons gene with random parameters."""
        num_cells = len(mutated_children) if mutated_children else 1
        tx_cell_index = get_rng().integers(0, num_cells)
        # Pick a random low-ish positive integer for rx_port (0-9)
        # The actual neuron will modulo this by its number of input ports when connecting
        rx_port = get_rng().integers(0, 10)
        new_connection = ConnectNeurons(tx_cell_index, rx_port, parent_gene=parent_gene)
        return [new_connection]

