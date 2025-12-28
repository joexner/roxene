from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ShuffleGenes(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "shuffle_genes"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def _mutate_CompositeGene_impl(self, parent_gene: CompositeGene) -> CompositeGene:
        # Only swap if there are at least 2 child genes
        if len(parent_gene.child_genes) < 2:
            return parent_gene
        
        # Get susceptibility for distance calculation
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        
        # Copy the genes list for swapping
        new_genes = list(parent_gene.child_genes)

        # Swap two genes - susceptibility influences maximum distance
        # Higher susceptibility = can swap genes that are further apart
        num_genes = len(new_genes)
        
        # Select first gene randomly
        first_index = get_rng().integers(0, num_genes)
        
        # Calculate max distance based on susceptibility
        # susceptibility near 0 -> only adjacent swaps
        # susceptibility near 1 -> can swap across entire list
        max_distance = max(1, int(susceptibility * num_genes))
        
        # Calculate possible second index range
        min_second = max(0, first_index - max_distance)
        max_second = min(num_genes - 1, first_index + max_distance)
        
        # Ensure we don't swap with self
        possible_indices = [i for i in range(min_second, max_second + 1) if i != first_index]
        
        if possible_indices:
            second_index = get_rng().choice(possible_indices)
            # Swap the genes
            new_genes[first_index], new_genes[second_index] = new_genes[second_index], new_genes[first_index]

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
