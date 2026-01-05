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

        min_second = max(0, first_index - max_distance)
        max_second = min(num_genes - 1, first_index + max_distance)
        possible_indices = [i for i in range(min_second, max_second + 1) if i != first_index]
        
        if possible_indices:
            second_index = get_rng().choice(possible_indices)
            new_genes[first_index], new_genes[second_index] = new_genes[second_index], new_genes[first_index]

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
