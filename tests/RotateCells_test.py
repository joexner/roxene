import unittest
from tests.ConnectNeurons_test import build_organism
from roxene.RotateCells import RotateCells


class RotateCells_test(unittest.TestCase):

    def test_execute(self):
        organism = build_organism(num_neurons=5, neuron_input_size=1)
        cells_before_execution = list(organism.cells)
        gene = RotateCells(1)
        gene.execute(organism)
        cells_after_execution =  list(organism.cells)
        self.assertEqual(cells_after_execution, cells_before_execution[-1:] + cells_before_execution[:-1])