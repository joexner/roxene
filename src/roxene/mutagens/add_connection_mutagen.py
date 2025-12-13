import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..genes.connect_neurons import ConnectNeurons
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class AddConnectionMutagen(Mutagen):
    """
    Adds a new ConnectNeurons gene to a CompositeGene.
    This creates new connections between cells in the organism.
    """
    __tablename__ = "add_connection_mutagen"
    __mapper_args__ = {"polymorphic_identity": "add_connection_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Create a new ConnectNeurons gene with random parameters
        num_cells = len(parent_gene.child_genes) if hasattr(parent_gene, "child_genes") else 1
        num_ports = getattr(parent_gene, "num_ports", 1)
        tx_cell_index = get_rng().integers(0, num_cells).astype(int)
        rx_port = get_rng().integers(0, num_ports).astype(int)
        new_connection = ConnectNeurons(tx_cell_index, rx_port, parent_gene=parent_gene)

        # Recursively mutate child genes
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)

        # Add the new connection gene
        new_genes.append(new_connection)
        any_changed = True

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
