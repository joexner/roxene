from abc import ABC
from numpy.random import Generator
from roxene import Gene
from roxene.genes import CompositeGene, CreateNeuron
from .util import wiggle


class Mutagen(ABC):
    susceptibilities: dict[Gene, float]
    susceptibility_log_wiggle: float

    def __init__(self, base_susceptibility: float, susceptibility_log_wiggle: float):
        self.susceptibilities = {None: base_susceptibility}
        self.susceptibility_log_wiggle = susceptibility_log_wiggle

    def get_mutation_susceptibility(self, gene: Gene, rng: Generator) -> float:
        result = self.susceptibilities.get(gene)
        if result is None:
            parent_gene = getattr(gene, "parent_gene", None)
            parent_sus = self.get_mutation_susceptibility(parent_gene, rng)
            result = wiggle(parent_sus, rng, self.susceptibility_log_wiggle)
            self.susceptibilities[gene] = result
        return result

    def mutate(self, gene: Gene, rng: Generator) -> Gene:
        if isinstance(gene, CompositeGene):
            return self.mutate_CompositeGene(gene, rng)
        elif isinstance(gene, CreateNeuron):
            return self.mutate_CreateNeuron(gene, rng)
        else:
            return gene

    def mutate_CompositeGene(self, parent_gene: CompositeGene, rng):
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig, rng)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)
        if any_changed:
            return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
        else:
            return parent_gene

    def mutate_CreateNeuron(self, gene: CreateNeuron, rng: Generator):
        return gene
