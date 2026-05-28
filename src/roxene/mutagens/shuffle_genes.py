from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ShuffleGenes(Mutagen):
    """
    Moves one child gene of a CompositeGene around in the execution order
    """

    __mapper_args__ = {"polymorphic_identity": "shuffle_genes"}


    def __init__(self, severity: float = 1.0, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.severity = severity


    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        if len(parent_gene.child_genes) < 2:
            return parent_gene
        
        num_genes = len(parent_gene.child_genes)
        
        # Pick a random gene to move
        source_index = get_rng().integers(num_genes)

        # Calculate valid destination range excluding source_index
        max_distance = max(1, int(self.severity * num_genes))
        min_dest = max(0, source_index - max_distance)
        max_dest = min(num_genes - 1, source_index + max_distance)
        valid_dests = [i for i in range(min_dest, max_dest + 1) if i != source_index]

        dest_index = valid_dests[get_rng().integers(0, len(valid_dests))]

        # Make a copy and move the gene from source to destination
        new_genes = list(parent_gene.child_genes)
        gene = new_genes.pop(source_index)
        new_genes.insert(dest_index, gene)
        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
