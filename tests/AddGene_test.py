import unittest
from operator import indexOf
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens.add_gene import AddGene
from roxene.util import set_rng

SEED = 456

new_gene = Gene()

# Concrete implementation for testing
class TestAddGene(AddGene):
    """Test implementation that adds a RotateCells gene"""
    __mapper_args__ = {"polymorphic_identity": "test_add_gene"}

    def get_new_gene(self, parent_gene: CompositeGene) -> Gene:
        return new_gene


class AddGene_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_correct_gene_count(self):
        """Test that insertion adds exactly one gene"""
        original_gene = CompositeGene(child_genes=[
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ])
        mutagen = TestAddGene()
        mutant_gene = mutagen.mutate_CompositeGene(original_gene)
        
        self.assertEqual(len(mutant_gene.child_genes), len(original_gene.child_genes) + 1)

    def test_empty_composite_gets_gene_added(self):
        """Test that empty CompositeGene gets a gene added to it"""
        original_gene = CompositeGene(child_genes=[])
        
        mutagen = TestAddGene()
        mutant_gene = mutagen.mutate_CompositeGene(original_gene)
        
        # Should now have 1 gene added
        self.assertEqual(len(mutant_gene.child_genes), 1)
        self.assertIs(mutant_gene.child_genes[0], new_gene)

    def test_insertion_index_range(self):
        """Test that insertion_index is always valid (0 to len inclusive)"""
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes)
        
        mutagen = TestAddGene()
        
        # Track which insertion indices are hit (0, 1, 2, 3 are valid for 3-element list)
        indices_hit = set()
        
        # Test many random insertions
        for seed in range(100):
            set_rng(default_rng(seed))
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            
            # Should always have exactly 4 genes (3 original + 1 inserted)
            self.assertEqual(len(mutant_gene.child_genes), 4)
            
            # Find position of inserted gene and record it
            indices_hit.add(indexOf(mutant_gene.child_genes, new_gene))

        # Assert all possible indices (0, 1, 2, 3) are covered
        self.assertEqual(indices_hit, {0, 1, 2, 3}, "All insertion indices should be covered")

    def test_preserves_parent_in_constructor(self):
        """Test that parent reference is passed to CompositeGene constructor"""
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.BACKWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=2)
        
        mutagen = TestAddGene()
        mutant_gene = mutagen.mutate_CompositeGene(original_gene)
        
        # Check the constructor was called with correct parameters
        self.assertEqual(mutant_gene.iterations, original_gene.iterations)
        self.assertEqual(len(mutant_gene.child_genes), len(original_gene.child_genes) + 1)
        self.assertEqual(mutant_gene.parent_gene, original_gene)
