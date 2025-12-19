import uuid
from enum import Enum, auto

import numpy as np
from sqlalchemy import Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng


class ResizeDirection(Enum):
    """Direction to resize the neuron's hidden layer."""
    WIDEN = auto()
    NARROW = auto()


class ResizeNeuronLayerMutagen(Mutagen):
    """
    Resizes the hidden layer of a CreateNeuron gene's neuron.
    Can widen by adding neurons or narrow by removing neurons from the hidden layer.
    """
    __tablename__ = "resize_neuron_layer_mutagen"
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)
    direction: Mapped[ResizeDirection] = mapped_column(SQLEnum(ResizeDirection))

    def __init__(self, 
                 direction: ResizeDirection,
                 base_susceptibility: float = 0.01, 
                 susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)
        self.direction = direction

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        if get_rng().random() >= susceptibility:
            return gene

        if self.direction == ResizeDirection.WIDEN:
            return self._widen_layer(gene)
        else:
            return self._narrow_layer(gene)

    def _widen_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Increases the size of the hidden layer by adding neurons."""
        # Get current hidden layer size from the shape of input_hidden
        current_hidden_size = gene.input_hidden.shape[1]
        
        # Increase hidden size by 1-3 neurons
        additional_neurons = get_rng().integers(1, 4)
        new_hidden_size = current_hidden_size + additional_neurons

        # Expand input_hidden: from [input_size, hidden_size] to [input_size, new_hidden_size]
        input_size = gene.input_hidden.shape[0]
        new_input_hidden = np.zeros([input_size, new_hidden_size], dtype=NP_PRECISION)
        new_input_hidden[:, :current_hidden_size] = gene.input_hidden
        new_input_hidden[:, current_hidden_size:] = (2 * get_rng().random([input_size, additional_neurons]) - 1).astype(NP_PRECISION)

        # Expand feedback_hidden: from [feedback_size, hidden_size] to [feedback_size, new_hidden_size]
        feedback_size = gene.feedback_hidden.shape[0]
        new_feedback_hidden = np.zeros([feedback_size, new_hidden_size], dtype=NP_PRECISION)
        new_feedback_hidden[:, :current_hidden_size] = gene.feedback_hidden
        new_feedback_hidden[:, current_hidden_size:] = (2 * get_rng().random([feedback_size, additional_neurons]) - 1).astype(NP_PRECISION)

        # Expand hidden_feedback: from [hidden_size, feedback_size] to [new_hidden_size, feedback_size]
        new_hidden_feedback = np.zeros([new_hidden_size, feedback_size], dtype=NP_PRECISION)
        new_hidden_feedback[:current_hidden_size, :] = gene.hidden_feedback
        new_hidden_feedback[current_hidden_size:, :] = (2 * get_rng().random([additional_neurons, feedback_size]) - 1).astype(NP_PRECISION)

        # Expand hidden_output: from [hidden_size, 1] to [new_hidden_size, 1]
        new_hidden_output = np.zeros([new_hidden_size, 1], dtype=NP_PRECISION)
        new_hidden_output[:current_hidden_size, :] = gene.hidden_output
        new_hidden_output[current_hidden_size:, :] = (2 * get_rng().random([additional_neurons, 1]) - 1).astype(NP_PRECISION)

        return CreateNeuron(
            input=gene.input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=new_input_hidden,
            hidden_feedback=new_hidden_feedback,
            feedback_hidden=new_feedback_hidden,
            hidden_output=new_hidden_output,
            parent_gene=gene
        )

    def _narrow_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Decreases the size of the hidden layer by removing neurons."""
        # Get current hidden layer size from the shape of input_hidden
        current_hidden_size = gene.input_hidden.shape[1]
        
        # Don't narrow if already at minimum size
        if current_hidden_size <= 1:
            return gene

        # Decrease hidden size by 1-2 neurons (but keep at least 1)
        max_remove = min(2, current_hidden_size - 1)
        neurons_to_remove = get_rng().integers(1, max_remove + 1)
        new_hidden_size = current_hidden_size - neurons_to_remove

        # Select random indices to keep
        indices_to_keep = get_rng().choice(current_hidden_size, new_hidden_size, replace=False)
        indices_to_keep = np.sort(indices_to_keep)

        # Narrow input_hidden: from [input_size, hidden_size] to [input_size, new_hidden_size]
        new_input_hidden = gene.input_hidden[:, indices_to_keep]

        # Narrow feedback_hidden: from [feedback_size, hidden_size] to [feedback_size, new_hidden_size]
        new_feedback_hidden = gene.feedback_hidden[:, indices_to_keep]

        # Narrow hidden_feedback: from [hidden_size, feedback_size] to [new_hidden_size, feedback_size]
        new_hidden_feedback = gene.hidden_feedback[indices_to_keep, :]

        # Narrow hidden_output: from [hidden_size, 1] to [new_hidden_size, 1]
        new_hidden_output = gene.hidden_output[indices_to_keep, :]

        return CreateNeuron(
            input=gene.input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=new_input_hidden,
            hidden_feedback=new_hidden_feedback,
            feedback_hidden=new_feedback_hidden,
            hidden_output=new_hidden_output,
            parent_gene=gene
        )
