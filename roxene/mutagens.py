import abc
from abc import ABC
from random import Random

import numpy as np
from numpy import sign, exp, log, ndarray
from numpy.random import Generator

from roxene import CreateNeuron, Gene
from roxene.constants import NP_PRECISION


class Mutagen(ABC):

    def __init__(self, severity: float, base_susceptibility: float, susceptibility_log_wiggle: float):
        self.severity = severity
        self.susceptibilities: dict[Gene, float] = {None: base_susceptibility}
        self.susceptibility_log_wiggle = susceptibility_log_wiggle

    @abc.abstractmethod
    def mutate(self, gene: Gene, rng: Random) -> Gene:
        pass

    def should_mutate(self, gene, rng):
        mutation_probability = self.get_mutation_susceptibility(gene, rng)
        return rng.random() < mutation_probability

    def get_mutation_susceptibility(self, gene: Gene, rng: Generator):
        result = self.susceptibilities.get(gene)
        if result is None:
            parent_gene = gene.parent_gene  # May be null, gets base_susceptibility
            parent_sus = self.get_mutation_susceptibility(parent_gene, rng)
            result = wiggle(parent_sus, rng, log_wiggle=self.susceptibility_log_wiggle)
            self.susceptibilities[gene] = result
        return result


class NeuronInitialValueMutagen(Mutagen):

    def __init__(self, severity: float = 0.5, base_susceptibility: float = 0.01,
                 susceptibility_log_wiggle: float = 0.01):
        super().__init__(severity, base_susceptibility, susceptibility_log_wiggle)

    def mutate(self, gene: CreateNeuron, rng: Generator) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene, rng)
        return CreateNeuron(
            input_initial_value=wiggle(gene.input_initial_value, rng, susceptibility * rng.random(),
                                       susceptibility * rng.random()).astype(NP_PRECISION),
            feedback_initial_value=wiggle(gene.feedback_initial_value, rng, susceptibility * rng.random(),
                                          susceptibility * rng.random()).astype(NP_PRECISION),
            output_initial_value=wiggle(gene.output_initial_value, rng, susceptibility * rng.random(),
                                        susceptibility * rng.random()).astype(NP_PRECISION),
            input_hidden=wiggle(gene.input_hidden, rng, susceptibility * rng.random(),
                                susceptibility * rng.random()).astype(NP_PRECISION),
            hidden_feedback=wiggle(gene.hidden_feedback, rng, susceptibility * rng.random(),
                                   susceptibility * rng.random()).astype(NP_PRECISION),
            feedback_hidden=wiggle(gene.feedback_hidden, rng, susceptibility * rng.random(),
                                   susceptibility * rng.random()).astype(NP_PRECISION),
            hidden_output=wiggle(gene.hidden_output, rng, susceptibility * rng.random(),
                                 susceptibility * rng.random()).astype(NP_PRECISION),
            parent_gene=gene
        )


def wiggle(x, rng: Generator, log_wiggle, absolute_wiggle=0):
    '''
    Randomly vary a value x != 0 by
    y = e^log(x +/- log_wiggle) +/- absolute_wiggle
    keeping the sign
    '''
    log_wiggled = sign(x) * exp(rng.normal(log(abs(x)), log_wiggle))
    return rng.normal(log_wiggled, absolute_wiggle)
