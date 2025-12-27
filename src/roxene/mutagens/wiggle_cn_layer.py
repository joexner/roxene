from enum import IntEnum, auto

import numpy as np
from numpy import ndarray
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class CNLayer(IntEnum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()


class WiggleCNLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "wiggle_cn_layer"}

    layer: Mapped[CNLayer] = synonym("_i1")

    def __init__(self, layer_to_mutate: CNLayer, base_susceptibility: float = 0.001,
                 susceptibility_log_wiggle: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.layer = layer_to_mutate
        self.severity = severity

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        return CreateNeuron(
            input=gene.input if self.layer != CNLayer.input_initial_value
            else self.maybe_wiggle(gene.input, susceptibility).astype(NP_PRECISION),
            feedback=gene.feedback if self.layer != CNLayer.feedback_initial_value
            else self.maybe_wiggle(gene.feedback, susceptibility).astype(NP_PRECISION),
            output=gene.output if self.layer != CNLayer.output_initial_value
            else self.maybe_wiggle(gene.output, susceptibility).astype(NP_PRECISION),
            input_hidden=gene.input_hidden if self.layer != CNLayer.input_hidden
            else self.maybe_wiggle(gene.input_hidden, susceptibility).astype(NP_PRECISION),
            hidden_feedback=gene.hidden_feedback if self.layer != CNLayer.hidden_feedback
            else self.maybe_wiggle(gene.hidden_feedback, susceptibility).astype(NP_PRECISION),
            feedback_hidden=gene.feedback_hidden if self.layer != CNLayer.feedback_hidden
            else self.maybe_wiggle(gene.feedback_hidden, susceptibility).astype(NP_PRECISION),
            hidden_output=gene.hidden_output if self.layer != CNLayer.hidden_output
            else self.maybe_wiggle(gene.hidden_output, susceptibility).astype(NP_PRECISION),
            parent_gene=gene
        )

    def maybe_wiggle(self, x: ndarray, susceptibility: float) -> ndarray:
        '''
        Use the susceptibility to derive the probability of mutating any given value in the mutated layer,
        and the log and absolute wiggles to use when mutating.
        Derive wiggle parameters from severity: log_wiggle = severity * 25, absolute_wiggle = severity * 1.0
        '''
        wiggle_probability = susceptibility
        log_wiggle = susceptibility * self.severity * 25.0
        absolute_wiggle = susceptibility * self.severity * 1.0
        return np.where(
            get_rng().random(x.shape) < wiggle_probability,
            x,
            wiggle(x, log_wiggle, absolute_wiggle))

