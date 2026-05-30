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

class SplitCompositeGene_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_basic(self):
        mutagen = SplitCompositeGene()
        for _ in range(20):
            original_gene = CompositeGene([RotateCells() for _ in range(3)], iterations=10)
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)
            self.assertIsInstance(mutant_gene, CompositeGene)
            self.assertEqual(mutant_gene.iterations, 1)
            self.assertEqual(len(mutant_gene.child_genes), 2)
            self.assertEqual(sum([child.iterations for child in mutant_gene.child_genes]), original_gene.iterations)
            for new_child_gene in mutant_gene.child_genes:
                self.assertIsInstance(new_child_gene, CompositeGene)
                self.assertSequenceEqual(new_child_gene.child_genes, original_gene.child_genes)
                self.assertGreaterEqual(new_child_gene.iterations, 1)

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