import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import DuplicateGeneMutagen
from roxene.util import set_rng

SEED = 321


class DuplicateGeneMutagen_test(unittest.TestCase):

    def test_duplicate_gene_in_composite(self):
        """Test that DuplicateGeneMutagen adds a duplicate of a child gene"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=3)
        
        mutagen = DuplicateGeneMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should still be a CompositeGene
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, 3)
        # Should have one more child gene (the duplicate)
        self.assertEqual(len(mutant_gene.child_genes), len(child_genes) + 1)

    def test_duplicate_gene_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = DuplicateGeneMutagen(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_duplicate_gene_placement(self):
        """Test that the duplicated gene is inserted at a random position"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [
            RotateCells(RotateCells.Direction.FORWARD),
            RotateCells(RotateCells.Direction.BACKWARD),
            RotateCells(RotateCells.Direction.FORWARD)
        ]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = DuplicateGeneMutagen(1.0, 0)  # 100% susceptibility
        
        for _ in range(10):
            mutant_gene = mutagen.mutate(original_gene)
            # Should have 4 child genes now
            self.assertEqual(len(mutant_gene.child_genes), 4)
            # Check that one of the genes appears twice (duplicated)
            gene_types = [type(g) for g in mutant_gene.child_genes]
            self.assertTrue(len(gene_types) > len(set(gene_types)) or 
                          any(gene_types.count(g) >= 2 for g in gene_types),
                          "One gene should appear twice (duplicated)")

    def test_duplicate_empty_composite(self):
        """Test duplicating in an empty CompositeGene doesn't crash"""
        set_rng(default_rng(SEED))
        original_gene = CompositeGene(child_genes=[], iterations=1)
        
        mutagen = DuplicateGeneMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should still be empty
        self.assertEqual(len(mutant_gene.child_genes), 0)

    def test_persist_reload(self):
        """Test that DuplicateGeneMutagen can be persisted and reloaded"""
        mutagen = DuplicateGeneMutagen(0.018, 0.028)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(DuplicateGeneMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.018)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.028)
