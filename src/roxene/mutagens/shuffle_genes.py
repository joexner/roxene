from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ShuffleGenes(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "shuffle_genes"}


    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        if len(parent_gene.child_genes) < 2:
            return parent_gene
        
        new_genes = list(parent_gene.child_genes)
        num_genes = len(new_genes)
        
        first_index = get_rng().integers(0, num_genes)
        
        max_distance = max(1, int(self.severity * num_genes))

        # Calculate valid range for second index
        min_second = max(0, first_index - max_distance)
        max_second = min(num_genes - 1, first_index + max_distance)
        
        # Number of valid positions excluding first_index
        num_valid = max_second - min_second  # range size minus 1 (for first_index)
        
        if num_valid > 0:
            # Pick a random offset in [0, num_valid), then map to actual index
            offset = get_rng().integers(0, num_valid)
            second_index = min_second + offset
            if second_index >= first_index:
                second_index += 1  # Skip over first_index
            new_genes[first_index], new_genes[second_index] = new_genes[second_index], new_genes[first_index]

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
