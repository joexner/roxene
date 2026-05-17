import unittest
from typing import List

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import RemoveGene
from roxene.util import set_rng

SEED = 11235


class RemoveGene_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))


    def test_basic(self):
        mutagen = RemoveGene()
        original_gene = CompositeGene(child_genes=[RotateCells(), RotateCells(), RotateCells()], iterations=2)
        mutant_gene = mutagen.mutate_CompositeGene(original_gene)
        self.assertIsInstance(mutant_gene, CompositeGene)
        self.assertEqual(mutant_gene.iterations, original_gene.iterations)
        self.assertEqual(len(mutant_gene.child_genes), len(original_gene.child_genes) - 1)


    def test_multiple_times(self):
        mutagen = RemoveGene()
        gene = CompositeGene([RotateCells() for _ in range(5)])

        # Keep mutating until it gets and stays empty
        for expected_count in [4, 3, 2, 1, 0, 0]:
            gene = mutagen.mutate_CompositeGene(gene)
            self.assertEqual(len(gene.child_genes), expected_count)


    def test_removes_random_gene(self):
        mutagen = RemoveGene()
        original_gene = CompositeGene([RotateCells() for _ in range(10)])
        ever_removed = set()

        for attempt in range(1000):
            result = mutagen.mutate_CompositeGene(original_gene)
            remaining = set(result.child_genes)
            self.assertTrue(remaining.issubset(original_gene.child_genes))
            removed = set(original_gene.child_genes) - remaining
            self.assertEqual(len(removed), 1)
            ever_removed.update(removed)
            if len(ever_removed) == len(original_gene.child_genes):
                print(f"All found in {attempt} attempts")
                break

        # All original child genes should have been removed at some point
        self.assertEqual(ever_removed, set(original_gene.child_genes))


    def test_persist_reload(self):
        """Test that RemoveGene can be persisted and reloaded"""
        mutagen = RemoveGene(0.015)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(RemoveGene, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.015)
