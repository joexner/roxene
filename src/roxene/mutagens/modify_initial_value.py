from enum import IntEnum, auto

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class InitialValueType(IntEnum):
    input = auto()
    feedback = auto()
    output = auto()


class ModifyInitialValue(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_initial_value"}

    layer: Mapped[InitialValueType] = synonym("_i1")

    def __init__(self, value_type: InitialValueType, base_susceptibility: float = 0.01,
                 susceptibility_log_wiggle: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.layer = value_type
        self.severity = severity

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        
        if self.layer == InitialValueType.input:
            modified_values = self._modify_values(gene.input, susceptibility)
            return CreateNeuron(
                input=modified_values,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        elif self.layer == InitialValueType.feedback:
            modified_values = self._modify_values(gene.feedback, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=modified_values,
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        elif self.layer == InitialValueType.output:
            modified_values = self._modify_values(gene.output, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=modified_values,
                input_hidden=gene.input_hidden,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        
        return gene

    def _modify_values(self, values: np.ndarray, susceptibility: float) -> np.ndarray:
        """
        Modify initial values surgically based on susceptibility.
        Each value has a probability of being modified based on susceptibility.
        """
        # Create a mask of which values to modify
        modify_mask = get_rng().random(values.shape) < susceptibility
        
        # Wiggle the selected values
        # Derive wiggle parameters from severity: log_wiggle = severity * 15, absolute_wiggle = severity * 0.3
        modified = values.copy()
        for idx in np.ndindex(values.shape):
            if modify_mask[idx]:
                log_wiggle = susceptibility * self.severity * 15.0
                absolute_wiggle = susceptibility * self.severity * 0.3
                modified[idx] = wiggle(values[idx], log_wiggle, absolute_wiggle)
        
        return modified.astype(NP_PRECISION)
