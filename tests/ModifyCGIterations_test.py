import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ModifyCGIterations
from roxene.util import set_rng, get_rng

SEED = 185


class ModifyCGIterations_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_mutate(self):
        """Test that ModifyIterations changes iteration count"""
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.FORWARD)]
        mutagen = ModifyCGIterations(0.01)
        
        for _ in range(100):
            original_gene = CompositeGene(child_genes=child_genes, iterations=get_rng().integers(0, 10))
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            
            # Should still be a CompositeGene
            self.assertIsInstance(mutant_gene, CompositeGene)
            # Should have at least 0 iterations
            self.assertGreaterEqual(mutant_gene.iterations, 0)
            # Iterations should have changed
            self.assertNotEqual(mutant_gene.iterations, original_gene.iterations)
            # Child genes should be preserved
            self.assertEqual(mutant_gene.child_genes, original_gene.child_genes)
            # Parent gene should be the original gene
            self.assertIs(mutant_gene.parent_gene, original_gene)

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
