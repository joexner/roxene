from enum import IntEnum, auto

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng


class ResizeDirection(IntEnum):
    WIDEN = auto()
    NARROW = auto()


class LayerToResize(IntEnum):
    INPUT = auto()
    HIDDEN = auto()
    FEEDBACK = auto()


class ResizeNeuronLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer"}

    direction: Mapped[ResizeDirection] = synonym("_i1")
    layer: Mapped[LayerToResize] = synonym("_i2")

    def __init__(self, direction: ResizeDirection, layer_to_resize: LayerToResize,
                 base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.direction = direction
        self.layer = layer_to_resize

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        susceptibility = self.get_mutation_susceptibility(gene)
        if get_rng().random() >= susceptibility:
            return gene

        if self.layer == LayerToResize.INPUT:
            if self.direction == ResizeDirection.WIDEN:
                return self._widen_input_layer(gene)
            else:
                return self._narrow_input_layer(gene)
        elif self.layer == LayerToResize.HIDDEN:
            if self.direction == ResizeDirection.WIDEN:
                return self._widen_hidden_layer(gene)
            else:
                return self._narrow_hidden_layer(gene)
        else:  # FEEDBACK
            if self.direction == ResizeDirection.WIDEN:
                return self._widen_feedback_layer(gene)
            else:
                return self._narrow_feedback_layer(gene)

    def _widen_hidden_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Increases the size of the hidden layer by adding one neuron."""
        # Get current hidden layer size from the shape of input_hidden
        current_hidden_size = gene.input_hidden.shape[1]
        
        # Increase hidden size by 1 neuron
        additional_neurons = 1
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

    def _narrow_hidden_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Decreases the size of the hidden layer by removing one neuron."""
        # Get current hidden layer size from the shape of input_hidden
        current_hidden_size = gene.input_hidden.shape[1]
        
        # Don't narrow if already at minimum size
        if current_hidden_size <= 1:
            return gene

        # Decrease hidden size by 1 neuron
        neurons_to_remove = 1
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

    def _widen_input_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Increases the size of the input layer by adding one neuron."""
        # Get current input layer size
        current_input_size = gene.input_hidden.shape[0]
        
        # Increase input size by 1 neuron
        new_input_size = current_input_size + 1
        hidden_size = gene.input_hidden.shape[1]

        # Expand input initial value
        new_input = np.zeros(new_input_size, dtype=NP_PRECISION)
        new_input[:current_input_size] = gene.input
        new_input[current_input_size:] = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)

        # Expand input_hidden: from [input_size, hidden_size] to [new_input_size, hidden_size]
        new_input_hidden = np.zeros([new_input_size, hidden_size], dtype=NP_PRECISION)
        new_input_hidden[:current_input_size, :] = gene.input_hidden
        new_input_hidden[current_input_size:, :] = (2 * get_rng().random([1, hidden_size]) - 1).astype(NP_PRECISION)

        return CreateNeuron(
            input=new_input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=new_input_hidden,
            hidden_feedback=gene.hidden_feedback,
            feedback_hidden=gene.feedback_hidden,
            hidden_output=gene.hidden_output,
            parent_gene=gene
        )

    def _narrow_input_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Decreases the size of the input layer by removing one neuron."""
        # Get current input layer size
        current_input_size = gene.input_hidden.shape[0]
        
        # Don't narrow if already at minimum size
        if current_input_size <= 1:
            return gene

        # Decrease input size by 1 neuron
        new_input_size = current_input_size - 1

        # Select random indices to keep
        indices_to_keep = get_rng().choice(current_input_size, new_input_size, replace=False)
        indices_to_keep = np.sort(indices_to_keep)

        # Narrow input initial value
        new_input = gene.input[indices_to_keep]

        # Narrow input_hidden: from [input_size, hidden_size] to [new_input_size, hidden_size]
        new_input_hidden = gene.input_hidden[indices_to_keep, :]

        return CreateNeuron(
            input=new_input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=new_input_hidden,
            hidden_feedback=gene.hidden_feedback,
            feedback_hidden=gene.feedback_hidden,
            hidden_output=gene.hidden_output,
            parent_gene=gene
        )

    def _widen_feedback_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Increases the size of the feedback layer by adding one neuron."""
        # Get current feedback layer size
        current_feedback_size = gene.feedback_hidden.shape[0]
        
        # Increase feedback size by 1 neuron
        new_feedback_size = current_feedback_size + 1
        hidden_size = gene.feedback_hidden.shape[1]

        # Expand feedback initial value
        new_feedback = np.zeros(new_feedback_size, dtype=NP_PRECISION)
        new_feedback[:current_feedback_size] = gene.feedback
        new_feedback[current_feedback_size:] = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)

        # Expand feedback_hidden: from [feedback_size, hidden_size] to [new_feedback_size, hidden_size]
        new_feedback_hidden = np.zeros([new_feedback_size, hidden_size], dtype=NP_PRECISION)
        new_feedback_hidden[:current_feedback_size, :] = gene.feedback_hidden
        new_feedback_hidden[current_feedback_size:, :] = (2 * get_rng().random([1, hidden_size]) - 1).astype(NP_PRECISION)

        # Expand hidden_feedback: from [hidden_size, feedback_size] to [hidden_size, new_feedback_size]
        new_hidden_feedback = np.zeros([hidden_size, new_feedback_size], dtype=NP_PRECISION)
        new_hidden_feedback[:, :current_feedback_size] = gene.hidden_feedback
        new_hidden_feedback[:, current_feedback_size:] = (2 * get_rng().random([hidden_size, 1]) - 1).astype(NP_PRECISION)

        return CreateNeuron(
            input=gene.input,
            feedback=new_feedback,
            output=gene.output,
            input_hidden=gene.input_hidden,
            hidden_feedback=new_hidden_feedback,
            feedback_hidden=new_feedback_hidden,
            hidden_output=gene.hidden_output,
            parent_gene=gene
        )

    def _narrow_feedback_layer(self, gene: CreateNeuron) -> CreateNeuron:
        """Decreases the size of the feedback layer by removing one neuron."""
        # Get current feedback layer size
        current_feedback_size = gene.feedback_hidden.shape[0]
        
        # Don't narrow if already at minimum size
        if current_feedback_size <= 1:
            return gene

        # Decrease feedback size by 1 neuron
        new_feedback_size = current_feedback_size - 1

        # Select random indices to keep
        indices_to_keep = get_rng().choice(current_feedback_size, new_feedback_size, replace=False)
        indices_to_keep = np.sort(indices_to_keep)

        # Narrow feedback initial value
        new_feedback = gene.feedback[indices_to_keep]

        # Narrow feedback_hidden: from [feedback_size, hidden_size] to [new_feedback_size, hidden_size]
        new_feedback_hidden = gene.feedback_hidden[indices_to_keep, :]

        # Narrow hidden_feedback: from [hidden_size, feedback_size] to [hidden_size, new_feedback_size]
        new_hidden_feedback = gene.hidden_feedback[:, indices_to_keep]

        return CreateNeuron(
            input=gene.input,
            feedback=new_feedback,
            output=gene.output,
            input_hidden=gene.input_hidden,
            hidden_feedback=new_hidden_feedback,
            feedback_hidden=new_feedback_hidden,
            hidden_output=gene.hidden_output,
            parent_gene=gene
        )
