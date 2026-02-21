import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ModifyCGIterations
from roxene.util import set_rng

SEED = 987


class ModifyIterationsMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_modify_iterations(self):
        """Test that ModifyIterations changes iteration count"""
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=5)
        
        mutagen = ModifyCGIterations(0.01)
        
        # Try multiple times to see variation in iteration changes
        different_iterations_found = False
        for _ in range(20):
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            
            # Should still be a CompositeGene
            self.assertIsInstance(mutant_gene, CompositeGene)
            # Should have at least 0 iterations
            self.assertGreaterEqual(mutant_gene.iterations, 0)
            
            if mutant_gene.iterations != original_gene.iterations:
                different_iterations_found = True
        
        self.assertTrue(different_iterations_found, 
                       "Expected iterations to change in at least one of 20 tries")


    def test_modify_iterations_minimum_zero(self):
        """Test that iterations never go below 0"""
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=0)
        
        mutagen = ModifyCGIterations(0.01)
        
        # Try many times to ensure we never get negative iterations
        for _ in range(50):
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            self.assertGreaterEqual(mutant_gene.iterations, 0, 
                                  "Iterations should never be less than 0")

    def test_modify_iterations_range(self):
        """Test that iteration changes are reasonable"""
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=10)
        
        mutagen = ModifyCGIterations(0.01)
        
        increases = 0
        decreases = 0
        for _ in range(100):
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            if mutant_gene.iterations > original_gene.iterations:
                increases += 1
            elif mutant_gene.iterations < original_gene.iterations:
                decreases += 1
        
        # Should have both increases and decreases with roughly equal probability
        self.assertGreater(increases, 0, "Should see some iteration increases")
        self.assertGreater(decreases, 0, "Should see some iteration decreases")

    def test_persist_reload(self):
        """Test that ModifyIterations can be persisted and reloaded"""
        severity = 0.26
        base_susceptibility = 0.012
        mutagen = ModifyCGIterations(severity, base_susceptibility)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ModifyCGIterations, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.severity, severity)
            self.assertEqual(reloaded.base_susceptibility, base_susceptibility)
