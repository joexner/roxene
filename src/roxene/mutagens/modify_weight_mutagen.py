import uuid
from enum import Enum, auto

import numpy as np
from sqlalchemy import Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class WeightLayer(Enum):
    """Enum for the different weight matrices in a neuron."""
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()


class ModifyWeightMutagen(Mutagen):
    """
    Surgically modifies specific connection weights in a CreateNeuron gene.
    Targets a specific weight matrix and modifies individual weights.
    """
    __tablename__ = "modify_weight_mutagen"
    __mapper_args__ = {"polymorphic_identity": "modify_weight_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)
    weight_layer: Mapped[WeightLayer] = mapped_column(SQLEnum(WeightLayer))
    log_wiggle_multiplier: Mapped[float] = mapped_column(default=10.0)
    absolute_wiggle_multiplier: Mapped[float] = mapped_column(default=0.5)

    def __init__(self,
                 weight_layer: WeightLayer,
                 base_susceptibility: float = 0.01,
                 susceptibility_log_wiggle: float = 0.01,
                 log_wiggle_multiplier: float = 10.0,
                 absolute_wiggle_multiplier: float = 0.5):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.weight_layer = weight_layer
        self.log_wiggle_multiplier = log_wiggle_multiplier
        self.absolute_wiggle_multiplier = absolute_wiggle_multiplier

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        
        # Select which weight matrix to modify
        if self.weight_layer == WeightLayer.input_hidden:
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
        elif self.weight_layer == WeightLayer.hidden_feedback:
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
        elif self.weight_layer == WeightLayer.feedback_hidden:
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
        elif self.weight_layer == WeightLayer.hidden_output:
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
        modified = weights.copy()
        for idx in np.ndindex(weights.shape):
            if modify_mask[idx]:
                log_wiggle = susceptibility * self.log_wiggle_multiplier
                absolute_wiggle = susceptibility * self.absolute_wiggle_multiplier
                modified[idx] = wiggle(weights[idx], log_wiggle, absolute_wiggle)
        
        return modified.astype(NP_PRECISION)
