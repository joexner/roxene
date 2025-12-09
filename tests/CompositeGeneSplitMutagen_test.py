import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import CompositeGeneSplitMutagen
from roxene.util import set_rng

SEED = 11235

class CompositeGeneSplitMutagen_test(unittest.TestCase):

    def test_split_basic(self):
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.BACKWARD),
                                   RotateCells(RotateCells.Direction.FORWARD),
                                   RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=10)

        mutagen = CompositeGeneSplitMutagen(1,0)

        set_rng(default_rng(SEED))
        for _ in range(20):
            mutant_gene = mutagen.mutate(original_gene)

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
        mutagen = CompositeGeneSplitMutagen(0.01, 0.03)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(CompositeGeneSplitMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.01)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.03)
