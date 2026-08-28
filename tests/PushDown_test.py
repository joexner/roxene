import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import PushDown
from roxene.util import set_rng

SEED = 42


class PushDownMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_basic(self):
        original_gene = RotateCells()
        mutagen = PushDown(1.0)
        mutant = mutagen.mutate(original_gene)
        self.assertIsInstance(mutant, CompositeGene)
        self.assertEqual(mutant.iterations, 1)
        self.assertEqual(len(mutant.child_genes), 1)
        self.assertEqual(mutant.child_genes[0], original_gene)

    def test_mutate_composite_gene(self):
        original_child_gene = RotateCells()
        original_gene = CompositeGene(child_genes=[original_child_gene], iterations=5)
        mutagen = PushDown(1.0)
        mutant = mutagen.mutate(original_gene)
        self.assertIsInstance(mutant, CompositeGene)
        self.assertEqual(mutant.iterations, 1)
        self.assertEqual(len(mutant.child_genes), 1)
        # The original gene, kinda
        mutant_child = mutant.child_genes[0]
        self.assertIsInstance(mutant_child, CompositeGene)
        self.assertEqual(mutant_child.iterations, 5)
        # New wrapper for the original child gene
        mutant_grandchild = mutant_child.child_genes[0]
        self.assertIsInstance(mutant_grandchild, CompositeGene)
        # Original child gene is 3 levels deep now
        self.assertIs(mutant_grandchild.child_genes[0], original_child_gene)

    def test_dont_mutate_single_iter_coposite(self):
        child_gene = RotateCells(RotateCells.Direction.FORWARD)
        original_gene = CompositeGene(child_genes=[child_gene], iterations=1)
        mutagen = PushDown(1.0)  # 100% susceptibility
        mutant = mutagen.mutate(original_gene)
        # Should not be wrapped - still has iterations=1
        self.assertIsInstance(mutant, CompositeGene)
        self.assertEqual(mutant.iterations, 1)
        # The child should be wrapped (since it's a RotateCells, not a composite with iterations=1)
        self.assertEqual(len(mutant.child_genes), 1)
        child = mutant.child_genes[0]
        self.assertIsInstance(child, CompositeGene)  # Child was wrapped

    def test_persist_reload(self):
        """Test that PushDown can be persisted and reloaded"""
        mutagen = PushDown(0.02)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(PushDown, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.02)
