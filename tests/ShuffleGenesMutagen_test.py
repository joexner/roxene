import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ShuffleGenesMutagen
from roxene.util import set_rng

SEED = 654


class ShuffleGenesMutagen_test(unittest.TestCase):

    def test_shuffle_genes_in_composite(self):
        """Test that ShuffleGenesMutagen changes the order of child genes"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=2)
        
        mutagen = ShuffleGenesMutagen(1.0, 0)  # 100% susceptibility
        
        # Try multiple times to ensure shuffle happens
        shuffled_found = False
        for _ in range(20):
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should still be a CompositeGene with same number of genes
            self.assertIsInstance(mutant_gene, CompositeGene)
            self.assertEqual(mutant_gene.iterations, 2)
            self.assertEqual(len(mutant_gene.child_genes), len(child_genes))
            
            # Check if order changed
            if mutant_gene.child_genes != child_genes:
                shuffled_found = True
                break
        
        self.assertTrue(shuffled_found, "Expected genes to be shuffled in at least one of 20 tries")

    def test_shuffle_genes_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenesMutagen(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_shuffle_genes_preserves_all_genes(self):
        """Test that shuffling preserves all genes (no additions or removals)"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenesMutagen(1.0, 0)  # 100% susceptibility
        
        for _ in range(10):
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should have same genes (possibly in different order)
            self.assertEqual(len(mutant_gene.child_genes), len(child_genes))
            self.assertEqual(sorted(mutant_gene.child_genes, key=id), 
                           sorted(child_genes, key=id))

    def test_shuffle_single_gene(self):
        """Test that CompositeGenes with only 1 child are not shuffled"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ShuffleGenesMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Can't shuffle a single gene, should remain the same
        self.assertEqual(len(mutant_gene.child_genes), 1)

    def test_persist_reload(self):
        """Test that ShuffleGenesMutagen can be persisted and reloaded"""
        mutagen = ShuffleGenesMutagen(0.022, 0.033)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ShuffleGenesMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.022)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.033)
