from ..genes.connect_neurons import ConnectNeurons
from ..mutagen import Mutagen
from ..util import get_rng


class RetargetConnectNeurons(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "retarget_connect_neurons"}


    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_ConnectNeurons(self, gene: ConnectNeurons) -> ConnectNeurons:
        max_delta = max(1, int(self.severity * 10))
        tx_delta = int(get_rng().integers(-max_delta, max_delta + 1))
        new_tx_index = max(0, gene.tx_cell_index + tx_delta)
        return ConnectNeurons(new_tx_index, gene.rx_port, parent_gene=gene)
