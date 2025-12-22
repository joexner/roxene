import uuid
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene
from ..util import get_rng


class DuplicateGene(AddGene):
    """
    Duplicates a random child gene in a CompositeGene.
    Inserts the duplicate at a random position in the gene list.
    """
    __tablename__ = "duplicate_gene_mutagen"
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("add_gene_mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def get_genes_to_insert(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> List[Gene]:
        """
        Returns a copy of a randomly selected child gene to duplicate.
        
        Args:
            parent_gene: The original CompositeGene being mutated
            mutated_children: The list of child genes after recursive mutation
            
        Returns:
            A list containing the duplicated gene, or empty list if no children
        """
        if len(mutated_children) == 0:
            return []
        
        # Select a random gene to duplicate
        index_to_duplicate = get_rng().integers(0, len(mutated_children))
        return [mutated_children[index_to_duplicate]]

