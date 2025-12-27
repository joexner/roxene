from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class RemoveGene(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "remove_gene"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        # Only remove if there are at least 2 child genes
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility or len(parent_gene.child_genes) < 2:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Remove a random gene first, before recursing
        new_genes = list(parent_gene.child_genes)
        index_to_remove = get_rng().integers(0, len(new_genes))
        new_genes.pop(index_to_remove)

        # Create intermediate gene with removed child, then delegate to base class to recursively mutate
        intermediate_gene = CompositeGene(new_genes, parent_gene.iterations, parent_gene.parent_gene)
        mutated_gene = super().mutate_CompositeGene(intermediate_gene)

        return CompositeGene(mutated_gene.child_genes, mutated_gene.iterations, parent_gene)
