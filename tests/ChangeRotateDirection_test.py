import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import RotateCells, CompositeGene
from roxene.mutagens import ChangeRotateDirection
from roxene.util import set_rng

SEED = 890


class ChangeRotateDirection_test(unittest.TestCase):

    def test_change_forward_to_backward(self):
        """Test that ChangeRotateDirection changes FORWARD to BACKWARD"""
        set_rng(default_rng(SEED))
        original = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = ChangeRotateDirection(1.0)  # 100% susceptibility
        
        mutant = mutagen.mutate(original)
        
        self.assertIsInstance(mutant, RotateCells)
        self.assertEqual(mutant.direction, RotateCells.Direction.BACKWARD)

    def test_change_backward_to_forward(self):
        """Test that ChangeRotateDirection changes BACKWARD to FORWARD"""
        set_rng(default_rng(SEED))
        original = RotateCells(RotateCells.Direction.BACKWARD)
        
        mutagen = ChangeRotateDirection(1.0)  # 100% susceptibility
        
        mutant = mutagen.mutate(original)
        
        self.assertIsInstance(mutant, RotateCells)
        self.assertEqual(mutant.direction, RotateCells.Direction.FORWARD)

    def test_parent_gene_set(self):
        """Test that the mutated gene has its parent_gene set correctly"""
        set_rng(default_rng(SEED))
        original = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = ChangeRotateDirection(1.0)
        
        mutant = mutagen.mutate(original)
        
        self.assertEqual(mutant.parent_gene, original)

    def test_no_mutation_at_zero_susceptibility(self):
        """Test that ChangeRotateDirection doesn't mutate at 0% susceptibility"""
        set_rng(default_rng(SEED))
        original = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = ChangeRotateDirection(0.0)  # 0% susceptibility
        
        mutant = mutagen.mutate(original)
        
        # Should be the same object (no mutation)
        self.assertIs(mutant, original)

    def test_change_in_composite(self):
        """Test that RotateCells within CompositeGenes are mutated"""
        set_rng(default_rng(SEED))
        rotate1 = RotateCells(RotateCells.Direction.FORWARD)
        rotate2 = RotateCells(RotateCells.Direction.BACKWARD)
        composite = CompositeGene(child_genes=[rotate1, rotate2], iterations=1)
        
        mutagen = ChangeRotateDirection(1.0)  # 100% susceptibility
        
        mutant_composite = mutagen.mutate(composite)
        
        self.assertIsInstance(mutant_composite, CompositeGene)
        self.assertEqual(len(mutant_composite.child_genes), 2)
        # Both children should have flipped directions
        directions = [gene.direction for gene in mutant_composite.child_genes]
        self.assertEqual(directions, [RotateCells.Direction.BACKWARD, RotateCells.Direction.FORWARD])

    def test_persist_reload(self):
        """Test that ChangeRotateDirection can be persisted and reloaded"""
        mutagen = ChangeRotateDirection(0.05)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ChangeRotateDirection, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.05)
