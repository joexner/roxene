from enum import IntEnum, auto

from numpy import ndarray
from sqlalchemy.orm import Mapped, synonym

from ..constants import NP_PRECISION
from ..genes.create_neuron import CreateNeuron
from ..mutagen import Mutagen
from ..util import wiggle


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
        return CreateNeuron(
            input=gene.input if self.layer != CNLayer.input_initial_value
            else self._wiggle_array(gene.input).astype(NP_PRECISION),
            feedback=gene.feedback if self.layer != CNLayer.feedback_initial_value
            else self._wiggle_array(gene.feedback).astype(NP_PRECISION),
            output=gene.output if self.layer != CNLayer.output_initial_value
            else self._wiggle_array(gene.output).astype(NP_PRECISION),
            input_hidden=gene.input_hidden if self.layer != CNLayer.input_hidden
            else self._wiggle_array(gene.input_hidden).astype(NP_PRECISION),
            hidden_feedback=gene.hidden_feedback if self.layer != CNLayer.hidden_feedback
            else self._wiggle_array(gene.hidden_feedback).astype(NP_PRECISION),
            feedback_hidden=gene.feedback_hidden if self.layer != CNLayer.feedback_hidden
            else self._wiggle_array(gene.feedback_hidden).astype(NP_PRECISION),
            hidden_output=gene.hidden_output if self.layer != CNLayer.hidden_output
            else self._wiggle_array(gene.hidden_output).astype(NP_PRECISION),
            parent_gene=gene
        )

    def _wiggle_array(self, x: ndarray) -> ndarray:
        log_wiggle = self.severity * 25.0
        absolute_wiggle = self.severity * 1.0
        return wiggle(x, log_wiggle, absolute_wiggle)

