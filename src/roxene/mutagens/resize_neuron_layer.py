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


def widen_layer(arr: np.ndarray, axis: Optional[int], insert_idx: int) -> np.ndarray:
    """Insert a new random value/slice at insert_idx, preserving all existing values."""
    new_shape = [1] if axis is None else [arr.shape[i] if i != axis else 1 for i in range(arr.ndim)]
    new_slice = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
    return np.insert(arr, insert_idx, new_slice.squeeze() if axis is None else new_slice, axis=axis)


def narrow_layer(arr: np.ndarray, axis: Optional[int], keep_indices: np.ndarray) -> np.ndarray:
    """Remove a value/slice by keeping only the specified indices."""
    return arr[keep_indices] if axis is None else np.take(arr, keep_indices, axis=axis)


def widen_layers(gene: "CreateNeuron", layer: "LayerToResize") -> Dict[str, np.ndarray]:
    """Widen all arrays for the specified layer. Calculates insert_idx once and reuses it."""
    layer_vec = getattr(gene, _SIZE_ATTR[layer])
    current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
    insert_idx = get_rng().integers(0, current_size + 1)
    
    result = {}
    for attr_name, axis in _LAYER_RESIZE_SPEC[layer]:
        result[attr_name] = widen_layer(getattr(gene, attr_name), axis, insert_idx)
    return result


def narrow_layers(gene: "CreateNeuron", layer: "LayerToResize") -> Dict[str, np.ndarray]:
    """Narrow all arrays for the specified layer. Calculates keep_indices once and reuses them."""
    layer_vec = getattr(gene, _SIZE_ATTR[layer])
    current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
    keep_indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
    
    result = {}
    for attr_name, axis in _LAYER_RESIZE_SPEC[layer]:
        result[attr_name] = narrow_layer(getattr(gene, attr_name), axis, keep_indices)
    return result


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
        
        # Get resized arrays using helpers that calculate indices once
        resized = widen_layers(gene, self.layer) if widen else narrow_layers(gene, self.layer)
        
        # Build kwargs for CreateNeuron, copying unchanged attributes and applying resized ones
        kwargs = {
            "input": gene.input, "feedback": gene.feedback, "output": gene.output,
            "input_hidden": gene.input_hidden, "hidden_feedback": gene.hidden_feedback,
            "feedback_hidden": gene.feedback_hidden, "hidden_output": gene.hidden_output,
            "parent_gene": gene
        }
        kwargs.update(resized)
        
        return CreateNeuron(**kwargs)
