from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class PushDown(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "push_down"}


    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate(self, gene: Gene) -> Gene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate if CompositeGene
            return super().mutate(gene)

        # Don't wrap CompositeGenes with iterations == 1, as that would be redundant
        if isinstance(gene, CompositeGene) and gene.iterations == 1:
            return super().mutate(gene)

        # Wrap the gene in a CompositeGene with a single iteration
        return CompositeGene(
            child_genes=[gene],
            iterations=1,
            parent_gene=gene
        )
