from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyIterations(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_iterations"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # First recurse into children via base class
        parent_gene = super().mutate_CompositeGene(parent_gene)
        
        # Check susceptibility before modifying iterations
        if not self.should_mutate(parent_gene):
            return parent_gene
        
        # Modify the iteration count by incrementing or decrementing by 1
        if get_rng().random() < 0.5:
            # Increment by 1
            new_iterations = parent_gene.iterations + 1
        else:
            # Decrement by 1, but never go below 0
            new_iterations = max(0, parent_gene.iterations - 1)

        # Return new gene with modified iterations
        return CompositeGene(parent_gene.child_genes, new_iterations, parent_gene)
