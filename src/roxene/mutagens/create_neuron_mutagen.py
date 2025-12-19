import uuid
from enum import Enum, auto

import numpy as np
from numpy import ndarray
from sqlalchemy import Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class CNLayer(Enum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()

class CreateNeuronMutagen(Mutagen):
    __tablename__ = "create_neuron_mutagen"
    __mapper_args__ = {"polymorphic_identity": "create_neuron_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)
    layer_to_mutate: Mapped[CNLayer] = mapped_column(SQLEnum(CNLayer))
    log_wiggle_multiplier: Mapped[float] = mapped_column(default=25.0)
    absolute_wiggle_multiplier: Mapped[float] = mapped_column(default=1.0)

    def __init__(self,
                 layer_to_mutate: CNLayer,
                 base_susceptibility: float = 0.001,
                 susceptibility_log_wiggle: float = 0.01,
                 log_wiggle_multiplier: float = 25.0,
                 absolute_wiggle_multiplier: float = 1.0):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.layer_to_mutate = layer_to_mutate
        self.log_wiggle_multiplier = log_wiggle_multiplier
        self.absolute_wiggle_multiplier = absolute_wiggle_multiplier

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        return CreateNeuron(
            input=gene.input if self.layer_to_mutate is not CNLayer.input_initial_value
            else self.maybe_wiggle(gene.input, susceptibility).astype(NP_PRECISION),
            feedback=gene.feedback if self.layer_to_mutate is not CNLayer.feedback_initial_value
            else self.maybe_wiggle(gene.feedback, susceptibility).astype(NP_PRECISION),
            output=gene.output if self.layer_to_mutate is not CNLayer.output_initial_value
            else self.maybe_wiggle(gene.output, susceptibility).astype(NP_PRECISION),
            input_hidden=gene.input_hidden if self.layer_to_mutate is not CNLayer.input_hidden
            else self.maybe_wiggle(gene.input_hidden, susceptibility).astype(NP_PRECISION),
            hidden_feedback=gene.hidden_feedback if self.layer_to_mutate is not CNLayer.hidden_feedback
            else self.maybe_wiggle(gene.hidden_feedback, susceptibility).astype(NP_PRECISION),
            feedback_hidden=gene.feedback_hidden if self.layer_to_mutate is not CNLayer.feedback_hidden
            else self.maybe_wiggle(gene.feedback_hidden, susceptibility).astype(NP_PRECISION),
            hidden_output=gene.hidden_output if self.layer_to_mutate is not CNLayer.hidden_output
            else self.maybe_wiggle(gene.hidden_output, susceptibility).astype(NP_PRECISION),
            parent_gene=gene
        )

    def maybe_wiggle(self, x: ndarray, susceptibility: float) -> ndarray:
        '''
        Use the susceptibility to derive the probability of mutating any given value in the mutated layer,
        and the log and absolute wiggles to use when mutating
        '''
        wiggle_probability = susceptibility
        log_wiggle = susceptibility * self.log_wiggle_multiplier
        absolute_wiggle = susceptibility * self.absolute_wiggle_multiplier
        return np.where(
            get_rng().random(x.shape) < wiggle_probability,
            x,
            wiggle(x, log_wiggle, absolute_wiggle))

