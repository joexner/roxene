import abc
from typing import List

from ..gene import Gene
from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class AddGene(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "add_gene"}


    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Recursively mutate child genes
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)

        # If there are no children, can't add a gene - return as-is
        if len(new_genes) == 0:
            return CompositeGene(new_genes, parent_gene.iterations, parent_gene)

        # Get the gene to insert - subclasses must implement this
        gene_to_insert = self.get_new_gene(parent_gene, new_genes)
        
        # Insert the gene at a random position
        # Choose a random index between 0 and len(new_genes) inclusive
        insertion_index = get_rng().integers(0, len(new_genes) + 1)
        new_genes.insert(insertion_index, gene_to_insert)

        return CompositeGene(new_genes, parent_gene.iterations, parent_gene)

    @abc.abstractmethod
    def get_new_gene(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> Gene:
        """
        Return the gene to insert into the CompositeGene.
        
        Args:
            parent_gene: The original CompositeGene being mutated
            mutated_children: The list of child genes after recursive mutation
            
        Returns:
            A gene to insert
        """
        pass

