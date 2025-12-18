import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class RemoveGeneMutagen(Mutagen):
    """
    Removes a random child gene from a CompositeGene.
    Only removes if the CompositeGene has more than one child gene.
    """
    __tablename__ = "remove_gene_mutagen"
    __mapper_args__ = {"polymorphic_identity": "remove_gene_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Only remove if there are at least 2 child genes
        if len(parent_gene.child_genes) < 2:
            return super().mutate_CompositeGene(parent_gene)

        # Delegate to base class to recursively mutate child genes
        mutated_gene = super().mutate_CompositeGene(parent_gene)

        # Remove a random gene from the mutated children
        new_genes = list(mutated_gene.child_genes)
        index_to_remove = get_rng().integers(0, len(new_genes))
        new_genes.pop(index_to_remove)

        return CompositeGene(new_genes, mutated_gene.iterations, parent_gene)
