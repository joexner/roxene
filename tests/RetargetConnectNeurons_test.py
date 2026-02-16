import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import ConnectNeurons, CompositeGene
from roxene.mutagens import RetargetConnectNeurons
from roxene.util import set_rng

SEED = 456


class RetargetConnectNeuronsMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_retarget_connect_neurons(self):
        """Test that RetargetConnectNeurons changes tx_index but not rx_port"""
        gene = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        mutagen = RetargetConnectNeurons()
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(gene)
            self.assertIsInstance(mutant, ConnectNeurons)
            self.assertNotEqual(mutant.tx_cell_index, gene.tx_cell_index)
            self.assertEqual(mutant.rx_port, gene.rx_port)

    def test_retarget_bounds(self):
        mutagen = RetargetConnectNeurons(severity = 0.25)
        original_gene = ConnectNeurons(tx_cell_index = 100, rx_input_port=5)
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(original_gene)
            tx_cell_delta = abs(mutant.tx_cell_index - original_gene.tx_cell_index)
            self.assertGreater(tx_cell_delta, 0)
            self.assertLessEqual(tx_cell_delta, original_gene.tx_cell_index * mutagen.severity)

