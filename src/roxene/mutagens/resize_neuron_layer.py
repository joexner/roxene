from enum import IntEnum, auto
from typing import Tuple

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng


class LayerToResize(IntEnum):
    INPUT = auto()
    HIDDEN = auto()
    FEEDBACK = auto()


def _expand_array(arr: np.ndarray, axis: int) -> np.ndarray:
    """Expand array by 1 along given axis, filling with random values in [-1, 1]."""
    shape = list(arr.shape)
    shape[axis] += 1
    new_arr = np.zeros(shape, dtype=NP_PRECISION)
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(arr.shape[axis])
    new_arr[tuple(slices)] = arr
    # Fill the new slice with random values
    slices[axis] = slice(arr.shape[axis], None)
    new_shape = list(shape)
    new_shape[axis] = 1
    new_arr[tuple(slices)] = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
    return new_arr


def _expand_vector(vec: np.ndarray) -> np.ndarray:
    """Expand 1D vector by 1, filling with random value in [-1, 1]."""
    new_vec = np.zeros(len(vec) + 1, dtype=NP_PRECISION)
    new_vec[:len(vec)] = vec
    new_vec[len(vec):] = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)
    return new_vec


def _narrow_array(arr: np.ndarray, axis: int, indices: np.ndarray) -> np.ndarray:
    """Narrow array along given axis, keeping only specified indices."""
    return np.take(arr, indices, axis=axis)


class ResizeNeuronLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer"}

    layer: Mapped[LayerToResize] = synonym("_i1")

    def __init__(self, layer_to_resize: LayerToResize,
                 base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.layer = layer_to_resize

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        widen = get_rng().random() < 0.5
        
        if self.layer == LayerToResize.INPUT:
            return self._resize_input(gene, widen)
        elif self.layer == LayerToResize.HIDDEN:
            return self._resize_hidden(gene, widen)
        else:  # FEEDBACK
            return self._resize_feedback(gene, widen)

    def _resize_input(self, gene: CreateNeuron, widen: bool) -> CreateNeuron:
        """Resize the input layer by 1 neuron."""
        current_size = gene.input_hidden.shape[0]
        if not widen and current_size <= 1:
            return gene
        
        if widen:
            return CreateNeuron(
                input=_expand_vector(gene.input),
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=_expand_array(gene.input_hidden, axis=0),
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        else:
            indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
            return CreateNeuron(
                input=gene.input[indices],
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=_narrow_array(gene.input_hidden, 0, indices),
                hidden_feedback=gene.hidden_feedback,
                feedback_hidden=gene.feedback_hidden,
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )

    def _resize_hidden(self, gene: CreateNeuron, widen: bool) -> CreateNeuron:
        """Resize the hidden layer by 1 neuron."""
        current_size = gene.input_hidden.shape[1]
        if not widen and current_size <= 1:
            return gene
        
        if widen:
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=_expand_array(gene.input_hidden, axis=1),
                hidden_feedback=_expand_array(gene.hidden_feedback, axis=0),
                feedback_hidden=_expand_array(gene.feedback_hidden, axis=1),
                hidden_output=_expand_array(gene.hidden_output, axis=0),
                parent_gene=gene
            )
        else:
            indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback,
                output=gene.output,
                input_hidden=_narrow_array(gene.input_hidden, 1, indices),
                hidden_feedback=_narrow_array(gene.hidden_feedback, 0, indices),
                feedback_hidden=_narrow_array(gene.feedback_hidden, 1, indices),
                hidden_output=_narrow_array(gene.hidden_output, 0, indices),
                parent_gene=gene
            )

    def _resize_feedback(self, gene: CreateNeuron, widen: bool) -> CreateNeuron:
        """Resize the feedback layer by 1 neuron."""
        current_size = gene.feedback_hidden.shape[0]
        if not widen and current_size <= 1:
            return gene
        
        if widen:
            return CreateNeuron(
                input=gene.input,
                feedback=_expand_vector(gene.feedback),
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=_expand_array(gene.hidden_feedback, axis=1),
                feedback_hidden=_expand_array(gene.feedback_hidden, axis=0),
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
        else:
            indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
            return CreateNeuron(
                input=gene.input,
                feedback=gene.feedback[indices],
                output=gene.output,
                input_hidden=gene.input_hidden,
                hidden_feedback=_narrow_array(gene.hidden_feedback, 1, indices),
                feedback_hidden=_narrow_array(gene.feedback_hidden, 0, indices),
                hidden_output=gene.hidden_output,
                parent_gene=gene
            )
