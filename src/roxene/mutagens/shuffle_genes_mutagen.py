import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ShuffleGenesMutagen(Mutagen):
    """
    Randomly shuffles the order of child genes in a CompositeGene.
    This can affect execution order and behavior.
    """
    __tablename__ = "shuffle_genes_mutagen"
    __mapper_args__ = {"polymorphic_identity": "shuffle_genes_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Only shuffle if there are at least 2 child genes
        if len(parent_gene.child_genes) < 2:
            return super().mutate_CompositeGene(parent_gene)

        # Recursively mutate child genes
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)

        # Shuffle the gene order
        get_rng().shuffle(new_genes)
        any_changed = True

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
