import logging
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy.orm import Mapped, synonym

from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import get_rng, random_slice

logger = logging.getLogger(__name__)

class LayerToResize(IntEnum):
    INPUT = auto()
    HIDDEN = auto()
    FEEDBACK = auto()


class ResizeDirection(IntEnum):
    WIDEN = auto()
    NARROW = auto()


# Config: maps layer -> [(array_attr, axis), ...]
# The first entry is used to determine the current layer size
LAYER_CONFIG: Dict[LayerToResize, List[Tuple[str, int]]] = {
    LayerToResize.INPUT: [("input", 0), ("input_hidden", 0)],
    LayerToResize.HIDDEN: [("input_hidden", 1), ("hidden_feedback", 0), ("feedback_hidden", 1), ("hidden_output", 0)],
    LayerToResize.FEEDBACK: [("feedback", 0), ("hidden_feedback", 1), ("feedback_hidden", 0)],
}


class ResizeNeuronLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "resize_neuron_layer"}

    layer: Mapped[LayerToResize] = synonym("_i1")
    direction: Mapped[ResizeDirection] = synonym("_i2")

    def __init__(self,
                 layer_to_resize: LayerToResize,
                 direction: ResizeDirection,
                 base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.layer = layer_to_resize
        self.direction = direction


    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        current_size = getattr(gene, LAYER_CONFIG[self.layer][0][0]).shape[LAYER_CONFIG[self.layer][0][1]]
        if current_size == 1 and self.direction == ResizeDirection.NARROW:
            logger.info(f"Skipping narrowing the {self.layer} layer because the layer is already 1")
            return gene
        mutant = CreateNeuron(
            input=gene.input,
            feedback=gene.feedback,
            output=gene.output,
            input_hidden=gene.input_hidden,
            hidden_feedback=gene.hidden_feedback,
            feedback_hidden=gene.feedback_hidden,
            hidden_output=gene.hidden_output,
            parent_gene=gene)
        if self.direction == ResizeDirection.WIDEN:
            idx = get_rng().integers(0, current_size + 1)
            for attr_name, axis in LAYER_CONFIG[self.layer]:
                arr = getattr(mutant, attr_name)
                new_shape = list(arr.shape)
                new_shape[axis] = 1
                new_slice = random_slice(new_shape).squeeze(axis=axis)
                resized = np.insert(arr, idx, new_slice, axis=axis)
                setattr(mutant, attr_name, resized)
        else:
            idx = get_rng().integers(0, current_size)
            for attr_name, axis in LAYER_CONFIG[self.layer]:
                arr = getattr(mutant, attr_name)
                resized = np.delete(arr, idx, axis=axis)
                setattr(mutant, attr_name, resized)
        return mutant
