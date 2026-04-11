from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class RemoveGene(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "remove_gene"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        if len(parent_gene.child_genes) == 0:
            return parent_gene

        new_genes = list(parent_gene.child_genes)
        index_to_remove = get_rng().integers(0, len(new_genes))
        new_genes.pop(index_to_remove)

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
