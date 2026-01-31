import unittest

from roxene.genes import ConnectNeurons, CompositeGene
from roxene.mutagens import AddConnectNeurons


class AddConnectNeurons_test(unittest.TestCase):

    def test_basic(self):
        gene = CompositeGene(child_genes=[], iterations=1)
        tx_cell_index = 3
        rx_port = 5
        mutagen = AddConnectNeurons(0.01, tx_cell_index=tx_cell_index, rx_port=rx_port)
        mutant = mutagen.get_new_gene(gene)
        self.assertIsInstance(mutant, ConnectNeurons)
        self.assertEqual(mutant.tx_cell_index, tx_cell_index)
        self.assertEqual(mutant.rx_port, rx_port)
