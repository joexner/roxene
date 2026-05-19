import unittest

from numpy.random import default_rng

from roxene.genes import ConnectNeurons
from roxene.mutagens import RetargetConnectNeurons
from roxene.util import set_rng

SEED = 11235


class RetargetConnectNeuronsMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_basic(self):
        mutagen = RetargetConnectNeurons()
        gene = ConnectNeurons(tx_cell_index=5, rx_input_port=3)
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(gene)
            self.assertIsInstance(mutant, ConnectNeurons)
            self.assertNotEqual(mutant.tx_cell_index, gene.tx_cell_index)
            self.assertEqual(mutant.rx_port, gene.rx_port)

    def test_bounds(self):
        mutagen = RetargetConnectNeurons(0.25)
        original_gene = ConnectNeurons(tx_cell_index = 100, rx_input_port=5)
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(original_gene)
            tx_cell_delta = abs(mutant.tx_cell_index - original_gene.tx_cell_index)
            self.assertGreater(tx_cell_delta, 0)
            self.assertLessEqual(tx_cell_delta, original_gene.tx_cell_index * mutagen.severity)

    def test_small_severity(self):
        mutagen = RetargetConnectNeurons(severity=0.1)
        original_gene = ConnectNeurons(tx_cell_index=100, rx_input_port=5)
        max_delta = int(mutagen.severity * original_gene.tx_cell_index)  # int(0.1 * 100) = 10
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(original_gene)
            tx_cell_delta = abs(mutant.tx_cell_index - original_gene.tx_cell_index)
            self.assertGreater(tx_cell_delta, 0)
            self.assertLessEqual(tx_cell_delta, max_delta)

    def test_clamp_to_one(self):
        mutagen = RetargetConnectNeurons(severity=1.0)
        original_gene = ConnectNeurons(tx_cell_index=1, rx_input_port=5)
        for _ in range(100):
            mutant = mutagen.mutate_ConnectNeurons(original_gene)
            self.assertGreaterEqual(mutant.tx_cell_index, 1)

