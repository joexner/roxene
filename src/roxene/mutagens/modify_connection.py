from ..genes.connect_neurons import ConnectNeurons
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyConnection(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_connection_mutagen"}


    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_ConnectNeurons(self, gene: ConnectNeurons) -> ConnectNeurons:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(gene)
        if get_rng().random() < susceptibility:
            # Mutate the connection parameters
            # Small random changes to the indices
            tx_delta = get_rng().integers(-2, 3)  # -2 to +2
            rx_delta = get_rng().integers(-2, 3)  # -2 to +2
            
            new_tx_index = max(0, gene.tx_cell_index + tx_delta)
            new_rx_port = max(0, gene.rx_port + rx_delta)
            
            return ConnectNeurons(new_tx_index, new_rx_port, parent_gene=gene)
        return gene
