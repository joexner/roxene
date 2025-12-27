from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyIterations(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_iterations"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Modify the iteration count by incrementing or decrementing by 1
        if get_rng().random() < 0.5:
            # Increment by 1
            new_iterations = parent_gene.iterations + 1
        else:
            # Decrement by 1, but never go below 0
            new_iterations = max(0, parent_gene.iterations - 1)

        # Create a new gene with modified iterations, then recurse via superclass
        modified_gene = CompositeGene(parent_gene.child_genes, new_iterations, parent_gene)
        return super().mutate_CompositeGene(modified_gene)
