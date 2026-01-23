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


def widen_layer(arr: np.ndarray, insert_idx: int, axis: Optional[int] = None) -> np.ndarray:
    """Insert a new random value/slice at insert_idx, preserving all existing values."""
    if axis is None:  # 1D vector
        new_val = (2 * get_rng().random(1) - 1).astype(NP_PRECISION)
        return np.insert(arr, insert_idx, new_val)
    else:
        new_shape = list(arr.shape)
        new_shape[axis] = 1
        new_slice = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
        return np.insert(arr, insert_idx, new_slice, axis=axis)


def narrow_layer(arr: np.ndarray, indices: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Remove a value/slice by keeping only the specified indices."""
    if axis is None:  # 1D vector
        return arr[indices]
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
        layer_vec = getattr(gene, _SIZE_ATTR[self.layer])
        current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
        
        if not widen and current_size <= 1:
            return gene
        
        # For widening: pick insertion position; for narrowing: pick indices to keep
        insert_idx = get_rng().integers(0, current_size + 1) if widen else 0
        indices = None if widen else np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
        
        # Build kwargs for CreateNeuron, copying all attributes
        kwargs = {
            "input": gene.input, "feedback": gene.feedback, "output": gene.output,
            "input_hidden": gene.input_hidden, "hidden_feedback": gene.hidden_feedback,
            "feedback_hidden": gene.feedback_hidden, "hidden_output": gene.hidden_output,
            "parent_gene": gene
        }
        
        # Apply resize to specified arrays
        for attr_name, axis in _LAYER_RESIZE_SPEC[self.layer]:
            arr = getattr(gene, attr_name)
            if widen:
                kwargs[attr_name] = widen_layer(arr, insert_idx, axis)
            else:
                kwargs[attr_name] = narrow_layer(arr, indices, axis)
        
        return CreateNeuron(**kwargs)
