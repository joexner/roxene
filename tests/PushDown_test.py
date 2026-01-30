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

    def test_push_down_basic(self):
        """Test that PushDown wraps a simple (non-composite) gene in a CompositeGene.
        
        A RotateCells gene should be wrapped in a CompositeGene with iterations=1.
        The original gene becomes the only child of the new composite.
        """
        original_gene = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = PushDown(1.0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should be wrapped in a CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 1)
        self.assertEqual(len(mutant_gene.child_genes), 1)
        self.assertEqual(mutant_gene.child_genes[0], original_gene)

    def test_push_down_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs.
        
        When base_susceptibility is 0.0, the gene should be returned unchanged.
        """
        original_gene = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = PushDown(0.0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_push_down_composite_gene_iterations_not_1(self):
        """Test that CompositeGenes with iterations != 1 get wrapped.
        
        A CompositeGene with iterations > 1 should be wrapped in a new 
        CompositeGene with iterations=1. The inner gene retains its original
        iterations count (5 in this test).
        """
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=5)
        
        mutagen = PushDown(1.0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should be wrapped in another CompositeGene with iterations=1
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 1)
        self.assertEqual(len(mutant_gene.child_genes), 1)
        # The wrapped child should be a CompositeGene with the original iterations
        wrapped_child = mutant_gene.child_genes[0]
        self.assertIsInstance(wrapped_child, CompositeGene)
        self.assertEqual(wrapped_child.iterations, 5)

    def test_push_down_composite_gene_iterations_1(self):
        """Test that CompositeGenes with iterations == 1 are NOT wrapped.
        
        A CompositeGene that already has iterations=1 should not be wrapped again 
        to avoid infinite nesting. Instead, the mutagen recursively applies to children.
        """
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = PushDown(1.0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be wrapped - still has iterations=1
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 1)
        # The child should be wrapped (since it's a RotateCells, not a composite with iterations=1)
        self.assertEqual(len(mutant_gene.child_genes), 1)
        child = mutant_gene.child_genes[0]
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
