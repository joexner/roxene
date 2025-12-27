from enum import IntEnum, auto

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class WeightLayer(IntEnum):
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()


class ModifyWeight(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_weight"}

    layer: Mapped[WeightLayer] = synonym("_i1")

    def __init__(self, weight_layer: WeightLayer, base_susceptibility: float = 0.01,
                 susceptibility_log_wiggle: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.layer = weight_layer
        self.severity = severity

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        
        if self.layer == WeightLayer.input_hidden:
            modified_weights = self._modify_weights(gene.input_hidden, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=modified_weights,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        elif self.layer == WeightLayer.hidden_feedback:
            modified_weights = self._modify_weights(gene.hidden_feedback, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=modified_weights,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        elif self.layer == WeightLayer.feedback_hidden:
            modified_weights = self._modify_weights(gene.feedback_hidden, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=modified_weights,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        elif self.layer == WeightLayer.hidden_output:
            modified_weights = self._modify_weights(gene.hidden_output, susceptibility)
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=modified_weights,
                parent_gene=gene
            )
        
        return gene

    def _modify_weights(self, weights: np.ndarray, susceptibility: float) -> np.ndarray:
        """
        Modify weights surgically based on susceptibility.
        Each weight has a probability of being modified based on susceptibility.
        """
        # Create a mask of which weights to modify
        modify_mask = get_rng().random(weights.shape) < susceptibility
        
        # Wiggle the selected weights
        # Derive wiggle parameters from severity: log_wiggle = severity * 10, absolute_wiggle = severity * 0.5
        modified = weights.copy()
        for idx in np.ndindex(weights.shape):
            if modify_mask[idx]:
                log_wiggle = susceptibility * self.severity * 10.0
                absolute_wiggle = susceptibility * self.severity * 0.5
                modified[idx] = wiggle(weights[idx], log_wiggle, absolute_wiggle)
        
        return modified.astype(NP_PRECISION)
