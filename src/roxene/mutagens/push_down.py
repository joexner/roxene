from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen


class PushDown(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "push_down"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate(self, gene: Gene) -> Gene:
        #TODO: Do this check in the Environment or something later
        # Check susceptibility before mutating
        if not self.should_mutate(gene):
            return super().mutate(gene)
            
        # Wrap the gene in a CompositeGene with a single iteration
        return CompositeGene(
            child_genes=[gene],
            iterations=1,
            parent_gene=gene
        )
