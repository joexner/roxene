from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen


class PushDown(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "push_down"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate(self, gene: Gene) -> Gene:
        mutant = super().mutate(gene)
        #TODO: Do this check in the Environment or something later
        # Check susceptibility before mutating
        if self.should_mutate(gene):
            # Wrap the gene in a CompositeGene with a single iteration
            mutant = CompositeGene(child_genes=[mutant], iterations=1, parent_gene=gene)
        return mutant

    def should_mutate(self, gene: Gene) -> bool:
        return isinstance(gene, CompositeGene) and gene.iterations == 1 and super().should_mutate(gene)
