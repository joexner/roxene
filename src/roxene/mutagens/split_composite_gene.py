from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class SplitCompositeGene(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "split_composite_gene"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        if parent_gene.iterations < 2:
            return super().mutate_CompositeGene(parent_gene)

        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Split the iterations randomly, ensuring both parts get at least 1 iteration
        # For iterations=N, we want first_iterations in range [1, N-1] so second_iterations is also ≥ 1
        first_iterations = get_rng().integers(1, parent_gene.iterations).astype(int)  # 1 to iterations-1 (exclusive upper bound)
        second_iterations = parent_gene.iterations - first_iterations

        # Create two new CompositeGenes with the same child genes but split iterations
        first_cg = CompositeGene(
            child_genes=parent_gene.child_genes,  # Same child genes
            iterations=first_iterations,
            parent_gene=parent_gene
        )

        second_cg = CompositeGene(
            child_genes=parent_gene.child_genes,  # Same child genes
            iterations=second_iterations,
            parent_gene=parent_gene
        )

        # Create a new parent CompositeGene containing the two split CompositeGenes
        return CompositeGene(
            child_genes=[first_cg, second_cg],
            iterations=1,  # Execute once to run both split genes
            parent_gene=parent_gene
        )
