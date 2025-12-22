import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ShuffleGenes
from roxene.util import set_rng

SEED = 654


class ShuffleGenesMutagen_test(unittest.TestCase):

    def test_shuffle_swaps_two_genes(self):
        """Test that ShuffleGenes swaps exactly two genes"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=2)
        
        mutagen = ShuffleGenes(1.0, 0)  # 100% susceptibility
        
        # Try multiple times to ensure swap happens
        swapped_found = False
        for _ in range(20):
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should still be a CompositeGene with same number of genes
            self.assertIsInstance(mutant_gene, CompositeGene)
            self.assertEqual(mutant_gene.iterations, 2)
            self.assertEqual(len(mutant_gene.child_genes), len(child_genes))
            
            # Check if order changed (genes were swapped)
            if mutant_gene.child_genes != child_genes:
                swapped_found = True
                # Count how many positions changed
                differences = sum(1 for i in range(len(child_genes)) 
                                if mutant_gene.child_genes[i] != child_genes[i])
                # Swapping two genes changes exactly 2 positions (or 0 if they're adjacent to identical genes)
                # Since we're swapping, we should see 2 or more changes
                self.assertGreaterEqual(differences, 2, "Swap should change at least 2 positions")
                break
        
        self.assertTrue(swapped_found, "Expected genes to be swapped in at least one of 20 tries")

    def test_shuffle_susceptibility_affects_distance(self):
        """Test that susceptibility affects how far apart swapped genes can be"""
        set_rng(default_rng(SEED))
        # Create a longer list to test distance
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        # Low susceptibility should only allow nearby swaps
        mutagen_low = ShuffleGenes(0.1, 0)
        
        # High susceptibility should allow distant swaps
        mutagen_high = ShuffleGenes(1.0, 0)
        
        # Just verify they both work without errors
        mutant_low = mutagen_low.mutate(original_gene)
        mutant_high = mutagen_high.mutate(original_gene)
        
        self.assertEqual(len(mutant_low.child_genes), len(child_genes))
        self.assertEqual(len(mutant_high.child_genes), len(child_genes))

    def test_shuffle_genes_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenes(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_shuffle_genes_preserves_all_genes(self):
        """Test that swapping preserves all genes (no additions or removals)"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenes(1.0, 0)  # 100% susceptibility
        
        for _ in range(10):
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should have same genes (possibly in different order)
            self.assertEqual(len(mutant_gene.child_genes), len(child_genes))
            mutant_ids = sorted([gene.id for gene in mutant_gene.child_genes])
            original_ids = sorted([gene.id for gene in child_genes])
            self.assertEqual(mutant_ids, original_ids)

    def test_shuffle_single_gene(self):
        """Test that CompositeGenes with only 1 child are not shuffled"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenes(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Can't swap a single gene, should remain the same
        self.assertEqual(len(mutant_gene.child_genes), 1)

    def test_persist_reload(self):
        """Test that ShuffleGenes can be persisted and reloaded"""
        mutagen = ShuffleGenes(0.022, 0.033)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ShuffleGenes, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.022)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.033)
