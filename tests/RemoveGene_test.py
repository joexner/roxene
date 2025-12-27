import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import RemoveGene
from roxene.util import set_rng

SEED = 789


class RemoveGeneMutagen_test(unittest.TestCase):

    def test_remove_gene_from_composite(self):
        """Test that RemoveGene removes a child gene"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=2)
        
        mutagen = RemoveGene(1.0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should still be a CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 2)
        # Should have one fewer child gene
        self.assertEqual(len(mutant_gene.child_genes), len(child_genes) - 1)

    def test_remove_gene_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = RemoveGene(0.0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_remove_gene_preserves_single_child(self):
        """Test that CompositeGenes with only 1 child are not reduced"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = RemoveGene(1.0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should still have 1 child (can't remove the last one)
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(len(mutant_gene.child_genes), 1)

    def test_remove_gene_multiple_times(self):
        """Test removing genes multiple times"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD)
        ]
        gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = RemoveGene(1.0)  # 100% susceptibility
        
        # Remove genes multiple times
        for expected_count in [4, 3, 2, 1, 1]:  # Can't go below 1
            gene = mutagen.mutate(gene)
            self.assertEqual(len(gene.child_genes), expected_count)

    def test_persist_reload(self):
        """Test that RemoveGene can be persisted and reloaded"""
        mutagen = RemoveGene(0.015)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(RemoveGene, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.015)
