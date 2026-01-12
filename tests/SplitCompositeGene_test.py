import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import SplitCompositeGene
from roxene.util import set_rng

SEED = 11235

class CompositeGeneSplitMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_split_basic(self):
        """
        Test that SplitCompositeGene correctly splits a CompositeGene into two children.
        
        When a CompositeGene with N iterations is split, the result should be a new
        CompositeGene with iterations=1 containing two child CompositeGenes. Each child
        should have the same child_genes as the original, and their iterations should
        sum to the original's iteration count.
        
        Example: A CompositeGene with 10 iterations might split into two CompositeGenes
        with 4 and 6 iterations respectively (the split point is random).
        """
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.BACKWARD),
                                   RotateCells(RotateCells.Direction.FORWARD),
                                   RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=10)

        mutagen = SplitCompositeGene(0.01)

        for _ in range(20):
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)

            self.assertIsInstance(mutant_gene, CompositeGene)
            self.assertEqual(mutant_gene.iterations, 1)
            self.assertEqual(len(mutant_gene.child_genes), 2)

            first, second = mutant_gene.child_genes

            self.assertIsInstance(first, CompositeGene)
            self.assertIsInstance(second, CompositeGene)
            self.assertSequenceEqual(first.child_genes, child_genes)
            self.assertSequenceEqual(second.child_genes, child_genes)
            self.assertEqual(first.iterations + second.iterations, 10)
            self.assertGreaterEqual(first.iterations, 1)
            self.assertGreaterEqual(second.iterations, 1)

    def test_persist_reload(self):
        mutagen = SplitCompositeGene(0.01)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(SplitCompositeGene, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.01)
