from ..genes.connect_neurons import ConnectNeurons
from ..mutagen import Mutagen
from ..util import get_rng


class RetargetConnectNeurons(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "retarget_connect_neurons"}


    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_ConnectNeurons(self, gene: ConnectNeurons) -> ConnectNeurons:
        # Max delta is severity * current target index, but at least 1
        max_delta = max(1, int(self.severity * gene.tx_cell_index))
        
        # Generate non-zero delta: choose from [-max_delta, max_delta] excluding 0
        rng = get_rng()
        tx_delta = int(rng.integers(1, max_delta + 1))  # 1 to max_delta
        if rng.random() < 0.5:
            tx_delta = -tx_delta  # 50% chance to be negative
        
        new_tx_index = max(0, gene.tx_cell_index + tx_delta)
        return ConnectNeurons(new_tx_index, gene.rx_port, parent_gene=gene)
