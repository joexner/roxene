import unittest

from roxene.genes import ConnectNeurons, CompositeGene
from roxene.mutagens import AddConnectNeurons


class AddConnectNeurons_test(unittest.TestCase):

    def test_get_new_gene_returns_connect_neurons(self):
        """
        Test that get_new_gene() returns a ConnectNeurons with the configured parameters.
        """
        mutagen = AddConnectNeurons(0.01, tx_cell_index=3, rx_port=5)
        parent = CompositeGene(child_genes=[], iterations=1)
        
        gene = mutagen.get_new_gene(parent)
        
        self.assertIsInstance(gene, ConnectNeurons)
        self.assertEqual(gene.tx_cell_index, 3)
        self.assertEqual(gene.rx_port, 5)
