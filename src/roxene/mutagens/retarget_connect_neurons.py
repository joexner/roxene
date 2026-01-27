from math import ceil

from ..genes.connect_neurons import ConnectNeurons
from ..mutagen import Mutagen
from ..util import get_rng


class RetargetConnectNeurons(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "retarget_connect_neurons"}


    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_ConnectNeurons(self, gene: ConnectNeurons) -> ConnectNeurons:
        # Low severity makes for small delta
        max_delta = max(1, int(self.severity * gene.tx_cell_index))
        sign = get_rng().choice([-1, 1])
        tx_delta = sign * get_rng().integers(1, max_delta + 1)
        new_tx_index = max(1, gene.tx_cell_index + tx_delta)
        return ConnectNeurons(new_tx_index, gene.rx_port, parent_gene=gene)
