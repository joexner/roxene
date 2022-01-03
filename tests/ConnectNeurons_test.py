from unittest import TestCase

import tensorflow as tf
from parameterized import parameterized

from roxene import ConnectNeurons, Neuron, Organism
from . import random_tensor

SEED = 8484856303


def connect_cells(organism, tx_idx, ports):
    for port in ports:
        ConnectNeurons(tx_idx, port).execute(organism)

class ConnectNeurons_test(TestCase):

    def setUp(self) -> None:
        tf.compat.v1.enable_eager_execution()
        tf.compat.v1.set_random_seed(SEED)

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
        self.assertIs(rx.input_ports[listener_port], expected_tx_cell)

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
        starting_port = 0

        organism = build_organism(num_neurons, neuron_input_size)
        rx_cell: Neuron = organism.cells[0]


        connect_cells(organism, 1, pre_connected_ports)
        num_connected_ports_before_connection = len(rx_cell.input_ports)

        self.assertEqual(num_connected_ports_before_connection, len(pre_connected_ports))


        gene = ConnectNeurons(tx_cell_index=1, rx_input_port=starting_port)

        for tx_cell_index in range(num_gene_executions):
            gene.execute(organism)

        num_connected_ports_after_connection = len(rx_cell.input_ports)

        self.assertEqual(num_connected_ports_after_connection, expected_num_connected_ports)


def build_organism(num_neurons=20, neuron_input_size=17):
    organism = Organism()
    for i in range(num_neurons):
        neuron = build_Neuron(neuron_input_size)
        organism.add(neuron)
    return organism

def build_Neuron(input_size):
    return Neuron(
        input_initial_value=random_tensor([input_size]),
        feedback_initial_value = random_tensor([1]),
        output_initial_value = random_tensor([]),
        input_hidden = random_tensor([1, 1]),
        hidden_feedback = random_tensor([1, 1]),
        feedback_hidden = random_tensor([1, 1]),
        hidden_output = random_tensor([1, 1])
    )
