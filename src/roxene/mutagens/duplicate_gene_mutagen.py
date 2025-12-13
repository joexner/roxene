import uuid
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from .insert_gene_to_composite_mutagen import InsertGeneToCompositeMutagen
from ..util import get_rng


class DuplicateGeneMutagen(InsertGeneToCompositeMutagen):
    """
    Duplicates a random child gene in a CompositeGene.
    Adds a copy of the selected gene to the gene list.
    """
    __tablename__ = "duplicate_gene_mutagen"
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("insert_gene_to_composite_mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self._index_to_duplicate = None

    def get_genes_to_insert(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> List[Gene]:
        """Select and duplicate a random gene from the mutated children."""
        if len(mutated_children) > 0:
            self._index_to_duplicate = get_rng().integers(0, len(mutated_children)).astype(int)
            return [mutated_children[self._index_to_duplicate]]
        return []

    def get_insertion_index(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> int:
        """Insert the duplicate right after the original."""
        if self._index_to_duplicate is not None:
            return self._index_to_duplicate + 1
        return 0


