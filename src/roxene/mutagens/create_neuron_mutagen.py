from enum import Enum, auto

import numpy as np
from numpy import ndarray
from numpy.random import Generator

from ..constants import NP_PRECISION
from ..genes import CreateNeuron
from .mutagen import Mutagen
from .util import wiggle

class CNLayer(Enum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()

class CreateNeuronMutagen(Mutagen):
    layer_to_mutate: CNLayer

    def __init__(self,
                 layer_to_mutate: CNLayer,
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

