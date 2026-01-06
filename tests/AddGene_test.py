import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens.add_gene import AddGene
from roxene.util import set_rng

SEED = 456


# Concrete implementation for testing
class TestAddGene(AddGene):
    """Test implementation that adds a RotateCells gene"""
    __mapper_args__ = {"polymorphic_identity": "test_add_gene"}
    
    def get_new_gene(self, parent_gene: CompositeGene, mutated_children: List[Gene]) -> Gene:
        return RotateCells(RotateCells.Direction.FORWARD)


class AddGene_test(unittest.TestCase):

    def test_correct_gene_count(self):
        """Test that insertion adds exactly one gene"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=3)
        
        mutagen = TestAddGene(1.0)
        mutant_gene = mutagen.mutate(original_gene)
        
        self.assertEqual(len(mutant_gene.child_genes), len(child_genes) + 1)
        self.assertEqual(mutant_gene.iterations, 3)

    def test_empty_composite_returns_unchanged(self):
        """Test that empty CompositeGene is returned unchanged"""
        set_rng(default_rng(SEED))
        original_gene = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = TestAddGene(1.0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should return original gene unchanged
        self.assertEqual(len(mutant_gene.child_genes), 0)
        self.assertIs(mutant_gene, original_gene)

    def test_insertion_index_range(self):
        """Test that insertion_index is always valid (0 to len inclusive)"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = TestAddGene(1.0)
        
        # Test many random insertions
        for seed in range(100):
            set_rng(default_rng(seed))
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should always have exactly 4 genes (3 original + 1 inserted)
            self.assertEqual(len(mutant_gene.child_genes), 4)
            
            # Should have exactly 1 FORWARD gene (the inserted one)
            forward_count = sum(1 for g in mutant_gene.child_genes 
                              if isinstance(g, RotateCells) and g.direction == RotateCells.Direction.FORWARD)
            self.assertEqual(forward_count, 1, "Should have exactly one inserted gene")

    def test_preserves_parent_in_constructor(self):
        """Test that parent reference is passed to CompositeGene constructor"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.BACKWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=2)
        
        mutagen = TestAddGene(1.0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Check the constructor was called with correct parameters
        self.assertEqual(mutant_gene.iterations, 2)
        self.assertEqual(len(mutant_gene.child_genes), 2)

    def test_uniform_distribution_of_insertions(self):
        """Test that insertions are roughly uniformly distributed across positions"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = TestAddGene(1.0)
        
        # Count insertions at each position
        position_counts = [0, 0, 0]  # positions 0, 1, 2 (before first, middle, after last)
        
        for seed in range(300):
            set_rng(default_rng(seed))
            mutant_gene = mutagen.mutate(original_gene)
            
            # Find position of inserted gene
            for i, g in enumerate(mutant_gene.child_genes):
                if isinstance(g, RotateCells) and g.direction == RotateCells.Direction.FORWARD:
                    position_counts[i] += 1
                    break
        
        # Each position should have roughly 100 insertions (with some variance)
        # Allow 60-140 range (within 2 standard deviations for uniform distribution)
        for i, count in enumerate(position_counts):
            self.assertGreater(count, 60, f"Position {i} should have at least 60 insertions, got {count}")
            self.assertLess(count, 140, f"Position {i} should have at most 140 insertions, got {count}")
