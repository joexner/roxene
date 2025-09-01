import unittest

from parameterized import parameterized
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import Neuron, Organism, random_neuron_state, EntityBase
from roxene.genes import ConnectNeurons

SEED = 8484856303



class ConnectNeurons_test(unittest.TestCase):

    @parameterized.expand([
        (2, 2),
        (2, 3),
        (1, 9),
        (50, 9),
    ])
    def test_execute(self, transmitter_idx, listener_port):
        organism = build_organism()
        gene = ConnectNeurons(transmitter_idx, listener_port)
        gene.execute(organism)
        rx: Neuron = organism.cells[0]
        expected_tx_cell = organism.cells[transmitter_idx % len(organism.cells)]
        self.assertIs(rx.bound_ports[listener_port], expected_tx_cell)

    @parameterized.expand([
        ([], 2, 2),
        ([0], 1, 2),
        ([0, 2, 4], 3, 6),
        ([0, 2, 5], 15, 10),
        ([0, 2, 6], 15, 10),
    ])
    def test_execute_repeated(self,
                              pre_connected_ports,
                              num_gene_executions,
                              expected_num_connected_ports
                              ):
        """Executing a ConnectNeurons gene handles an occupied port correctly

        That is, if a gene wants to connect to an occupied input port n, it will automatically connect to
        another unoccupied port if available
        """
        num_neurons = 2
        neuron_input_size = 10
        starting_port = 9

        organism = build_organism(num_neurons, neuron_input_size)

        rx_cell = organism.cells[0]
        tx_cell = organism.cells[1]

        for port in pre_connected_ports:
            rx_cell.add_input_connection(tx_cell, port)

        num_connected_ports_before_connection = len(rx_cell.bound_ports)

        self.assertEqual(num_connected_ports_before_connection, len(pre_connected_ports))


        gene = ConnectNeurons(tx_cell_index=1, rx_input_port=starting_port)

        for tx_cell_index in range(num_gene_executions):
            gene.execute(organism)

        num_connected_ports_after_connection = len(rx_cell.bound_ports)

        self.assertEqual(num_connected_ports_after_connection, expected_num_connected_ports)

    def test_negative_indices(self):
        """Negative indices should be handled correctly"""

        organism = build_organism(10, 10)
        gene = ConnectNeurons(-1, -2)
        gene.execute(organism)

        rx: Neuron = organism.cells[0]
        expected_tx_cell = organism.cells[9]
        self.assertIs(rx.bound_ports[8], expected_tx_cell)

    def test_persistence(self):

        gene = ConnectNeurons(1, 2)
        gene_id = gene.id

        gene_2 = ConnectNeurons(-3, -11)
        gene_2_id = gene_2.id

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            session.add_all([gene, gene_2])
            session.commit()

        with Session(engine) as session:
            reloaded = session.get(ConnectNeurons, gene_id)
            self.assertEqual(reloaded.tx_cell_index, 1)
            self.assertEqual(reloaded.rx_port, 2)

            reloaded_2 = session.get(ConnectNeurons, gene_2_id)
            self.assertEqual(reloaded_2.tx_cell_index, -3)
            self.assertEqual(reloaded_2.rx_port, -11)


def build_organism(num_neurons: int = 20, neuron_input_size: int = 17, input_names=set(), output_names=set(),
                   rng=None) -> Organism:
    organism = Organism(input_names=input_names, output_names=output_names)
    for i in range(num_neurons):
        neuron = build_Neuron(neuron_input_size, rng)
        organism.addNeuron(neuron)
    return organism


def build_Neuron(input_size, rng=None):
    return Neuron(**random_neuron_state(input_size, 1, 1, rng))
