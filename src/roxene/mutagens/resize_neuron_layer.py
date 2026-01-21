from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

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

# Which attribute to check for current size (use layer vectors directly where possible)
_SIZE_ATTR: Dict[LayerToResize, str] = {
    LayerToResize.INPUT: "input",
    LayerToResize.HIDDEN: "hidden_output",  # hidden size from hidden_output rows
    LayerToResize.FEEDBACK: "feedback",
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
        layer_vec = getattr(gene, _SIZE_ATTR[self.layer])
        current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
        
        if not widen and current_size <= 1:
            return gene
        
        # For narrowing: pick indices to keep; for widening: pick insertion position
        if widen:
            indices = None
            insert_idx = get_rng().integers(0, current_size + 1)
        else:
            indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
            insert_idx = 0
        
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
                kwargs[attr_name] = self._resize_vector(arr, widen, indices, insert_idx)
            else:
                kwargs[attr_name] = self._resize_array(arr, ax, widen, indices, insert_idx)
        
        return CreateNeuron(**kwargs)

    def _resize_vector(self, vec: np.ndarray, widen: bool, 
                       indices: Optional[np.ndarray], insert_idx: int = 0) -> np.ndarray:
        if widen:
            # Insert a new random value at insert_idx, preserving all existing values
            new_val = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)
            return np.insert(vec, insert_idx, new_val)
        return vec[indices]

    def _resize_array(self, arr: np.ndarray, axis: int, widen: bool,
                      indices: Optional[np.ndarray], insert_idx: int = 0) -> np.ndarray:
        if widen:
            # Insert a new random slice at insert_idx along axis, preserving all existing values
            new_shape = list(arr.shape)
            new_shape[axis] = 1
            new_slice = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
            return np.insert(arr, insert_idx, new_slice, axis=axis)
        return np.take(arr, indices, axis=axis)
