from enum import IntEnum, auto
from typing import Dict, List, Tuple, Optional

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


# Define which arrays to resize for each layer type: (attr_name, axis)
# For vectors (input, feedback), axis is None
_LAYER_RESIZE_SPEC: Dict[LayerToResize, List[Tuple[str, Optional[int]]]] = {
    LayerToResize.INPUT: [("input", None), ("input_hidden", 0)],
    LayerToResize.HIDDEN: [("input_hidden", 1), ("hidden_feedback", 0), 
                           ("feedback_hidden", 1), ("hidden_output", 0)],
    LayerToResize.FEEDBACK: [("feedback", None), ("hidden_feedback", 1), 
                             ("feedback_hidden", 0)],
}

# Which attribute/axis to check for current size
_SIZE_CHECK: Dict[LayerToResize, Tuple[str, int]] = {
    LayerToResize.INPUT: ("input_hidden", 0),
    LayerToResize.HIDDEN: ("input_hidden", 1),
    LayerToResize.FEEDBACK: ("feedback_hidden", 0),
}


class ResizeNeuronLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer"}

    layer: Mapped[LayerToResize] = synonym("_i1")

    def __init__(self, layer_to_resize: LayerToResize,
                 base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.layer = layer_to_resize

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        widen = get_rng().random() < 0.5
        attr, axis = _SIZE_CHECK[self.layer]
        current_size = getattr(gene, attr).shape[axis]
        
        if not widen and current_size <= 1:
            return gene
        
        indices = None if widen else np.sort(
            get_rng().choice(current_size, current_size - 1, replace=False))
        
        # Build kwargs for CreateNeuron, copying all attributes
        kwargs = {
            "input": gene.input, "feedback": gene.feedback, "output": gene.output,
            "input_hidden": gene.input_hidden, "hidden_feedback": gene.hidden_feedback,
            "feedback_hidden": gene.feedback_hidden, "hidden_output": gene.hidden_output,
            "parent_gene": gene
        }
        
        # Apply resize to specified arrays
        for attr_name, ax in _LAYER_RESIZE_SPEC[self.layer]:
            arr = getattr(gene, attr_name)
            if ax is None:  # 1D vector
                kwargs[attr_name] = self._resize_vector(arr, widen, indices)
            else:
                kwargs[attr_name] = self._resize_array(arr, ax, widen, indices)
        
        return CreateNeuron(**kwargs)

    def _resize_vector(self, vec: np.ndarray, widen: bool, 
                       indices: Optional[np.ndarray]) -> np.ndarray:
        if widen:
            new_vec = np.zeros(len(vec) + 1, dtype=NP_PRECISION)
            new_vec[:len(vec)] = vec
            new_vec[len(vec):] = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)
            return new_vec
        return vec[indices]

    def _resize_array(self, arr: np.ndarray, axis: int, widen: bool,
                      indices: Optional[np.ndarray]) -> np.ndarray:
        if widen:
            shape = list(arr.shape)
            shape[axis] += 1
            new_arr = np.zeros(shape, dtype=NP_PRECISION)
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(arr.shape[axis])
            new_arr[tuple(slices)] = arr
            slices[axis] = slice(arr.shape[axis], None)
            new_shape = list(shape)
            new_shape[axis] = 1
            new_arr[tuple(slices)] = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
            return new_arr
        return np.take(arr, indices, axis=axis)
