from unittest import TestCase
from unittest.mock import Mock

import tests
from roxene import Organism, Gene

SEED = 22049456


class Organism_test(TestCase):


    def test_constructor_cells(self):
        cells = [tests.make_neuron(2, 2, 2) for i in range(5)]
        organism_from_cells = Organism(cells=cells)
        self.assertSequenceEqual(organism_from_cells.cells, cells)

    def test_constructor_genoytpe(self):
        root_gene: Gene = Mock(Gene)
        organism: Organism = Organism(genoytpe=root_gene)
        print(root_gene.method_calls)
        root_gene.execute.assert_called_once_with(organism)

