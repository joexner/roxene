from enum import IntEnum, auto

from numpy import ndarray
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from .. import util


class CNLayer(IntEnum):
    input_initial_value = auto()
    feedback_initial_value = auto()
    output_initial_value = auto()
    input_hidden = auto()
    hidden_feedback = auto()
    feedback_hidden = auto()
    hidden_output = auto()

class WiggleCNLayer(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "wiggle_cn_layer"}

    layer: Mapped[CNLayer] = synonym("_i1")

    def __init__(self, layer_to_mutate: CNLayer, base_susceptibility: float = 0.001,
                 severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.layer = layer_to_mutate
        self.severity = severity

    def mutate_CreateNeuron(self, gene: CreateNeuron) -> CreateNeuron:
        args = {
            "input":           self.wiggle(gene.input)           if self.layer == CNLayer.input_initial_value    else gene.input,
            "feedback":        self.wiggle(gene.feedback)        if self.layer == CNLayer.feedback_initial_value else gene.feedback,
            "output":          self.wiggle(gene.output)          if self.layer == CNLayer.output_initial_value   else gene.output,
            "input_hidden":    self.wiggle(gene.input_hidden)    if self.layer == CNLayer.input_hidden           else gene.input_hidden,
            "hidden_feedback": self.wiggle(gene.hidden_feedback) if self.layer == CNLayer.hidden_feedback        else gene.hidden_feedback,
            "feedback_hidden": self.wiggle(gene.feedback_hidden) if self.layer == CNLayer.feedback_hidden        else gene.feedback_hidden,
            "hidden_output":   self.wiggle(gene.hidden_output)   if self.layer == CNLayer.hidden_output          else gene.hidden_output,
            "parent_gene":     gene
        }
        return CreateNeuron(**args)

    def wiggle(self, x: ndarray) -> ndarray:
        return util.wiggle(x, log_wiggle = self.severity * 25.0, absolute_wiggle = self.severity * 1.0)

