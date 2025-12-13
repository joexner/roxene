import uuid
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from .insert_gene_to_composite_mutagen import InsertGeneToCompositeMutagen
from ..util import get_rng


class DuplicateGeneMutagen(InsertGeneToCompositeMutagen):
    """
    Duplicates a random child gene in a CompositeGene.
    Adds a copy of the selected gene to the gene list.
    
    Note: This mutagen overrides mutate_CompositeGene directly to handle
    the special case where the duplicate must be inserted right after the
    original gene, requiring a single random index selection.
    """
    __tablename__ = "duplicate_gene_mutagen"
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("insert_gene_to_composite_mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes using base Mutagen behavior
            return Mutagen.mutate_CompositeGene(self, parent_gene)

        # Recursively mutate child genes
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)

        # Duplicate a random gene
        if len(new_genes) > 0:
            index_to_duplicate = get_rng().integers(0, len(new_genes)).astype(int)
            # Insert the duplicate right after the original
            new_genes.insert(index_to_duplicate + 1, new_genes[index_to_duplicate])

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)

    def get_genes_to_insert(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> List[Gene]:
        """
        Not used - this mutagen overrides mutate_CompositeGene directly.
        Required by abstract base class.
        """
        raise NotImplementedError("DuplicateGeneMutagen overrides mutate_CompositeGene directly")

    def get_insertion_index(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> int:
        """
        Not used - this mutagen overrides mutate_CompositeGene directly.
        Default implementation provided for completeness.
        """
        return len(mutated_children)

