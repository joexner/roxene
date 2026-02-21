from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng, random_slice


class LayerToResize(IntEnum):
    INPUT = auto()
    HIDDEN = auto()
    FEEDBACK = auto()


# Combined config: maps layer -> (size_attr, [(array_attr, axis), ...])
# size_attr: which vector to check for current layer size
# array specs: (attribute_name, axis) where axis=None means 1D vector
LAYER_CONFIG: Dict[LayerToResize, Tuple[str, List[Tuple[str, Optional[int]]]]] = {
    LayerToResize.INPUT: ("input", [("input", None), ("input_hidden", 0)]),
    LayerToResize.HIDDEN: ("hidden_output", [("input_hidden", 1), ("hidden_feedback", 0), ("feedback_hidden", 1), ("hidden_output", 0)]),
    LayerToResize.FEEDBACK: ("feedback", [("feedback", None), ("hidden_feedback", 1), ("feedback_hidden", 0)]),
}


def widen_layer(arr: np.ndarray, axis: Optional[int], insert_idx: int) -> np.ndarray:
    """Insert a new random value/slice at insert_idx, preserving all existing values."""
    new_shape = [1] if axis is None else [arr.shape[i] if i != axis else 1 for i in range(arr.ndim)]
    new_val = random_slice(new_shape)
    return np.insert(arr, insert_idx, new_val.squeeze() if axis is None else new_val, axis=axis)


def narrow_layer(arr: np.ndarray, axis: Optional[int], remove_idx: int) -> np.ndarray:
    """Remove a value/slice at the specified index."""
    return np.delete(arr, remove_idx, axis=axis)


def widen_layers(gene: "CreateNeuron", layer: "LayerToResize") -> Dict[str, np.ndarray]:
    """Widen all arrays for the specified layer. Calculates insert_idx once and reuses it."""
    size_attr, array_specs = LAYER_CONFIG[layer]
    layer_vec = getattr(gene, size_attr)
    current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
    insert_idx = get_rng().integers(0, current_size + 1)
    
    result = {}
    for attr_name, axis in array_specs:
        result[attr_name] = widen_layer(getattr(gene, attr_name), axis, insert_idx)
    return result


def narrow_layers(gene: "CreateNeuron", layer: "LayerToResize") -> Dict[str, np.ndarray]:
    """Narrow all arrays for the specified layer. Calculates remove_idx once and reuses it."""
    size_attr, array_specs = LAYER_CONFIG[layer]
    layer_vec = getattr(gene, size_attr)
    current_size = len(layer_vec) if layer_vec.ndim == 1 else layer_vec.shape[0]
    remove_idx = get_rng().integers(0, current_size)
    
    result = {}
    for attr_name, axis in array_specs:
        result[attr_name] = narrow_layer(getattr(gene, attr_name), axis, remove_idx)
    return result


class ResizeNeuronLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer"}

    layer: Mapped[LayerToResize] = synonym("_i1")

    def __init__(self, layer_to_resize: LayerToResize, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.layer = layer_to_resize

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        widen = get_rng().random() < 0.5
        size_attr, _ = LAYER_CONFIG[self.layer]
        layer_vec = getattr(gene, size_attr)
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
