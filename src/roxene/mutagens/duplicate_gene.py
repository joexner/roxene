from typing import List

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene
from ..util import get_rng


class DuplicateGene(AddGene):
    """
    Mutagen that duplicates an existing gene within a CompositeGene.
    That same exact Gene will be in 2 different spots in the CompositeGene, Via AddGene,
    and each spot in the new Composite can be mutated independently in descendants.
    """
    __mapper_args__ = {"polymorphic_identity": "duplicate_gene"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Can't duplicate from an empty CompositeGene
        if len(parent_gene.child_genes) == 0:
            return parent_gene
        return super().mutate_CompositeGene(parent_gene)

    def get_new_gene(self, parent_gene: CompositeGene) -> Gene:
        # Select a random gene to duplicate
        index_to_duplicate = get_rng().integers(0, len(parent_gene.child_genes))
        return parent_gene.child_genes[index_to_duplicate]

