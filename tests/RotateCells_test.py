import unittest
from parameterized import parameterized

from tests.ConnectNeurons_test import build_organism
from roxene.RotateCells import RotateCells


class RotateCells_test(unittest.TestCase):

    @parameterized.expand([
        (1,),
        (2,),
        (-1,),
        (4,),
        (7,),
    ])
    def test_execute(self, direction):
        num_neurons = 5
        organism = build_organism(num_neurons, neuron_input_size=1)
        cells_before_execution = list(organism.cells)
        gene = RotateCells(direction)
        gene.execute(organism)
        cells_after_execution =  list(organism.cells)
        cutoff = (-1 * direction) % num_neurons
        expected_cells_after_execution = cells_before_execution[cutoff:] + cells_before_execution[:cutoff]
        self.assertEqual(cells_after_execution, expected_cells_after_execution)