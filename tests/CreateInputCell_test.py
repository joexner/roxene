from random import Random

import tensorflow as tf

from roxene import Organism, InputCell
from roxene.genes import CreateInputCell

SEED = 84592184

class CreateInputCell_test(tf.test.TestCase):

    def test_execute(self):
        rng = Random(SEED)
        org = Organism()
        initial_value = rng.uniform(-1, 1)
        gene = CreateInputCell(initial_value)
        self.assertEmpty(org.cells)
        gene.execute(org)
        self.assertEqual(len(org.cells), 1)
        new_cell: InputCell = org.cells[0]
        self.assertEqual(new_cell.get_output(), initial_value)

