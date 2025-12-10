import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ModifyIterationsMutagen
from roxene.util import set_rng

SEED = 987


class ModifyIterationsMutagen_test(unittest.TestCase):

    def test_modify_iterations(self):
        """Test that ModifyIterationsMutagen changes iteration count"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=5)
        
        mutagen = ModifyIterationsMutagen(1.0, 0)  # 100% susceptibility
        
        # Try multiple times to see variation in iteration changes
        different_iterations_found = False
        for _ in range(20):
            mutant_gene = mutagen.mutate(original_gene)
            
            # Should still be a CompositeGene
            self.assertIsInstance(mutant_gene, CompositeGene)
            # Should have at least 1 iteration
            self.assertGreaterEqual(mutant_gene.iterations, 1)
            
            if mutant_gene.iterations != original_gene.iterations:
                different_iterations_found = True
        
        self.assertTrue(different_iterations_found, 
                       "Expected iterations to change in at least one of 20 tries")

    def test_modify_iterations_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=3)
        
        mutagen = ModifyIterationsMutagen(0.0, 0)  # 0% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_modify_iterations_minimum_one(self):
        """Test that iterations never go below 1"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=1)
        
        mutagen = ModifyIterationsMutagen(1.0, 0)  # 100% susceptibility
        
        # Try many times to ensure we never get 0 or negative
        for _ in range(50):
            mutant_gene = mutagen.mutate(original_gene)
            self.assertGreaterEqual(mutant_gene.iterations, 1, 
                                  "Iterations should never be less than 1")

    def test_modify_iterations_range(self):
        """Test that iteration changes are reasonable"""
        set_rng(default_rng(SEED))
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=10)
        
        mutagen = ModifyIterationsMutagen(1.0, 0)  # 100% susceptibility
        
        increases = 0
        decreases = 0
        for _ in range(100):
            mutant_gene = mutagen.mutate(original_gene)
            if mutant_gene.iterations > original_gene.iterations:
                increases += 1
            elif mutant_gene.iterations < original_gene.iterations:
                decreases += 1
        
        # Should have both increases and decreases with roughly equal probability
        self.assertGreater(increases, 0, "Should see some iteration increases")
        self.assertGreater(decreases, 0, "Should see some iteration decreases")

    def test_persist_reload(self):
        """Test that ModifyIterationsMutagen can be persisted and reloaded"""
        mutagen = ModifyIterationsMutagen(0.012, 0.019)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ModifyIterationsMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.012)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.019)
