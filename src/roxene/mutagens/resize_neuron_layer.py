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


def widen_layer(arr: np.ndarray, axis: Optional[int] = None, insert_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Insert a new random value/slice, preserving all existing values.
    Returns (new_array, insert_idx_used) so the same index can be reused for related arrays."""
    current_size = len(arr) if axis is None else arr.shape[axis]
    if insert_idx is None:
        insert_idx = get_rng().integers(0, current_size + 1)
    new_shape = [1] if axis is None else [arr.shape[i] if i != axis else 1 for i in range(arr.ndim)]
    new_slice = (2 * get_rng().random(new_shape) - 1).astype(NP_PRECISION)
    return np.insert(arr, insert_idx, new_slice.squeeze() if axis is None else new_slice, axis=axis), insert_idx


def narrow_layer(arr: np.ndarray, axis: Optional[int] = None, indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a value/slice by keeping only the specified indices.
    Returns (new_array, indices_used) so the same indices can be reused for related arrays."""
    current_size = len(arr) if axis is None else arr.shape[axis]
    if indices is None:
        indices = np.sort(get_rng().choice(current_size, current_size - 1, replace=False))
    return (arr[indices] if axis is None else np.take(arr, indices, axis=axis)), indices


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
        
        # Build kwargs for CreateNeuron, copying all attributes
        kwargs = {
            "input": gene.input, "feedback": gene.feedback, "output": gene.output,
            "input_hidden": gene.input_hidden, "hidden_feedback": gene.hidden_feedback,
            "feedback_hidden": gene.feedback_hidden, "hidden_output": gene.hidden_output,
            "parent_gene": gene
        }
        
        # Apply resize to specified arrays (first call calculates idx, subsequent reuse it)
        idx = None
        for attr_name, axis in _LAYER_RESIZE_SPEC[self.layer]:
            arr = getattr(gene, attr_name)
            if widen:
                kwargs[attr_name], idx = widen_layer(arr, axis, idx)
            else:
                kwargs[attr_name], idx = narrow_layer(arr, axis, idx)
        
        return CreateNeuron(**kwargs)
