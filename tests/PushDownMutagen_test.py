import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import PushDownMutagen
from roxene.util import set_rng

SEED = 42


class PushDownMutagen_test(unittest.TestCase):

    def test_push_down_basic(self):
        """Test that PushDownMutagen wraps a simple gene in a CompositeGene"""
        set_rng(default_rng(SEED))
        original_gene = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = PushDownMutagen(1.0, 0)  # 100% susceptibility, no wiggle
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should be wrapped in a CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 1)
        self.assertEqual(len(mutant_gene.child_genes), 1)
        self.assertEqual(mutant_gene.child_genes[0], original_gene)

    def test_push_down_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        original_gene = RotateCells(RotateCells.Direction.FORWARD)
        
        mutagen = PushDownMutagen(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_push_down_composite_gene_iterations_not_1(self):
        """Test that CompositeGenes with iterations != 1 get wrapped"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=5)
        
        mutagen = PushDownMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should be wrapped in another CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 1)
        self.assertEqual(len(mutant_gene.child_genes), 1)
        self.assertEqual(mutant_gene.child_genes[0], original_gene)

    def test_push_down_composite_gene_iterations_1(self):
        """Test that CompositeGenes with iterations == 1 are not wrapped"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = PushDownMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should recursively mutate but not wrap (since iterations == 1)
        self.assertIsInstance(mutant_gene, CompositeGene)
        # Should be same or mutated child genes, but not wrapped again
        self.assertEqual(mutant_gene.iterations, 1)

    def test_persist_reload(self):
        """Test that PushDownMutagen can be persisted and reloaded"""
        mutagen = PushDownMutagen(0.02, 0.05)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(PushDownMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.02)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.05)
