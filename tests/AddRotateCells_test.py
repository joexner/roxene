import unittest

from numpy.random import default_rng

from roxene.genes import RotateCells, CompositeGene
from roxene.mutagens import AddRotateCells
from roxene.util import set_rng

SEED = 789


class AddRotateCells_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_add_rotate_cells(self):
        """Test that AddRotateCells adds a RotateCells gene with the configured direction"""
        composite = CompositeGene(child_genes=[], iterations=1)
        
        # Test FORWARD direction
        mutagen_forward = AddRotateCells(0.01, RotateCells.Direction.FORWARD)
        mutant = mutagen_forward.mutate_CompositeGene(composite)
        self.assertEqual(len(mutant.child_genes), 1)
        self.assertIsInstance(mutant.child_genes[0], RotateCells)
        self.assertEqual(mutant.child_genes[0].direction, RotateCells.Direction.FORWARD)
        
        # Test BACKWARD direction
        mutagen_backward = AddRotateCells(0.01, RotateCells.Direction.BACKWARD)
        mutant = mutagen_backward.mutate_CompositeGene(composite)
        self.assertEqual(len(mutant.child_genes), 1)
        self.assertIsInstance(mutant.child_genes[0], RotateCells)
        self.assertEqual(mutant.child_genes[0].direction, RotateCells.Direction.BACKWARD)

    def test_add_rotate_cells_default_direction(self):
        """Test that AddRotateCells defaults to BACKWARD direction"""
        composite = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = AddRotateCells(0.01)  # No direction specified
        
        mutant_composite = mutagen.mutate_CompositeGene(composite)
        
        added_gene = mutant_composite.child_genes[0]
        self.assertEqual(added_gene.direction, RotateCells.Direction.BACKWARD)
