from typing import List

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene
from ..util import get_rng


class DuplicateGene(AddGene):
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene_mutagen"}


    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def get_new_gene(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> Gene:
        """
        Returns a copy of a randomly selected child gene to duplicate.
        
        Args:
            parent_gene: The original CompositeGene being mutated
            mutated_children: The list of child genes after recursive mutation
            
        Returns:
            The duplicated gene
        """
        # Select a random gene to duplicate
        index_to_duplicate = get_rng().integers(0, len(mutated_children))
        return mutated_children[index_to_duplicate]

