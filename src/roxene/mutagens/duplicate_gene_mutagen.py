import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class DuplicateGeneMutagen(Mutagen):
    """
    Duplicates a random child gene in a CompositeGene.
    Adds a copy of the selected gene to the gene list.
    """
    __tablename__ = "duplicate_gene_mutagen"
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Recursively mutate child genes
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)

        # Duplicate a random gene
        if len(new_genes) > 0:
            index_to_duplicate = get_rng().integers(0, len(new_genes)).astype(int)
            # Insert the duplicate right after the original
            new_genes.insert(index_to_duplicate + 1, new_genes[index_to_duplicate])
            any_changed = True

        if any_changed:
            return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
        else:
            return parent_gene
