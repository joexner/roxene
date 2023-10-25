from abc import ABC
from random import Random

import numpy as np
from numpy import sign, exp, log, ndarray
from numpy.random import Generator

from .constants import NP_PRECISION
from .genes import CreateNeuron, Gene, CompositeGene, CNLayer


class Mutagen(ABC):

    def __init__(self, base_susceptibility: float, susceptibility_log_wiggle: float):
        self.susceptibilities: dict[Gene, float] = {None: base_susceptibility}
        self.susceptibility_log_wiggle = susceptibility_log_wiggle

    def get_mutation_susceptibility(self, gene: Gene, rng: Generator) -> float:
        result = self.susceptibilities.get(gene)
        if result is None:
            parent_gene = gene.parent_gene  # May be null, gets base_susceptibility
            parent_sus = self.get_mutation_susceptibility(parent_gene, rng)
            result = wiggle(parent_sus, rng, self.susceptibility_log_wiggle)
            self.susceptibilities[gene] = result
        return result

    def mutate(self, gene: Gene, rng: Random) -> Gene:
        if isinstance(gene, CompositeGene):
            return self.mutate_CompositeGene(gene, rng)
        elif isinstance(gene, CreateNeuron):
            return self.mutate_CreateNeuron(gene, rng)
        else:
            return gene

    def mutate_CompositeGene(self, parent_gene: CompositeGene, rng):
        any_changed = False
        new_genes = []
        for orig in parent_gene.genes:
            mutant = self.mutate(orig, rng)
            new_genes.append(mutant)
            any_changed &= (mutant is not orig)
        if any_changed:
            return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
        else:
            return parent_gene

    def mutate_CreateNeuron(self, gene: CreateNeuron, rng: Generator):
        return gene


class CreateNeuronMutagen(Mutagen):
    layer_to_mutate: CNLayer

    def __init__(self, layer_to_mutate: CNLayer,
                 base_susceptibility: float = 0.001,
                 susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.layer_to_mutate = layer_to_mutate

    def mutate_CreateNeuron(self, gene: CreateNeuron, rng: Generator) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene, rng)
        return CreateNeuron(
            input=gene.input if self.layer_to_mutate is not CNLayer.input_initial_value
            else self.maybe_wiggle(gene.input, susceptibility, rng).astype(NP_PRECISION),
            feedback=gene.feedback if self.layer_to_mutate is not CNLayer.feedback_initial_value
            else self.maybe_wiggle(gene.feedback, susceptibility, rng).astype(NP_PRECISION),
            output=gene.output if self.layer_to_mutate is not CNLayer.output_initial_value
            else self.maybe_wiggle(gene.output, susceptibility, rng).astype(NP_PRECISION),
            input_hidden=gene.input_hidden if self.layer_to_mutate is not CNLayer.input_hidden
            else self.maybe_wiggle(gene.input_hidden, susceptibility, rng).astype(NP_PRECISION),
            hidden_feedback=gene.hidden_feedback if self.layer_to_mutate is not CNLayer.hidden_feedback
            else self.maybe_wiggle(gene.hidden_feedback, susceptibility, rng).astype(NP_PRECISION),
            feedback_hidden=gene.feedback_hidden if self.layer_to_mutate is not CNLayer.feedback_hidden
            else self.maybe_wiggle(gene.feedback_hidden, susceptibility, rng).astype(NP_PRECISION),
            hidden_output=gene.hidden_output if self.layer_to_mutate is not CNLayer.hidden_output
            else self.maybe_wiggle(gene.hidden_output, susceptibility, rng).astype(NP_PRECISION),
            parent_gene=gene
        )

    def maybe_wiggle(self, x: ndarray, susceptibility: float, rng: Generator) -> ndarray:
        '''
        Use the susceptibility to derive the probability of mutating any given value in the mutated layer,
        and the log and absolute wiggles to use when mutating
        '''
        wiggle_probability = susceptibility
        log_wiggle = susceptibility * 25
        absolute_wiggle = susceptibility
        return np.where(
            rng.random(x.shape) < wiggle_probability,
            x,
            wiggle(x, rng, log_wiggle, absolute_wiggle))


def wiggle(x, rng: Generator, log_wiggle, absolute_wiggle=0):
    '''
    Randomly vary a value x != 0 by
    y = e^log(x +/- log_wiggle) +/- absolute_wiggle
    keeping the sign
    '''
    log_wiggled = sign(x) * exp(rng.normal(log(abs(x)), log_wiggle))
    return rng.normal(log_wiggled, absolute_wiggle)
