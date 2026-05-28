import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import ShuffleGenes
from roxene.util import get_rng, set_rng

SEED = 11235


class ShuffleGenesMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))


    def test_basic(self):
        """Test that ShuffleGenes moves one gene to a new position"""
        for _ in range(20):
            mutagen = ShuffleGenes(severity=get_rng().random())
            original_gene = CompositeGene([RotateCells() for _ in range(10)], iterations=2)
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)

            # Should still be a CompositeGene with same genes in a different order
            self.assertIsInstance(mutant_gene, CompositeGene)
            self.assertEqual(mutant_gene.iterations, original_gene.iterations)
            self.assertEqual(len(mutant_gene.child_genes), len(original_gene.child_genes))
            self.assertSetEqual(set(mutant_gene.child_genes), set(original_gene.child_genes))

            # Order must change (a gene is always moved to a different position)
            self.assertNotEqual(mutant_gene.child_genes, original_gene.child_genes, "Shuffle should change gene order")

            # Find the gene that moved the furthest
            moved_gene = max(
                original_gene.child_genes,
                key=lambda g: abs(original_gene.child_genes.index(g) - mutant_gene.child_genes.index(g))
            )

            # Check that removing it from both lists leaves the same order
            orig_without_moved = [g for g in original_gene.child_genes if g is not moved_gene]
            mut_without_moved = [g for g in mutant_gene.child_genes if g is not moved_gene]
            self.assertEqual(mut_without_moved, orig_without_moved, "Exactly one gene should have moved")


    def test_shuffle_severity_affects_distance(self):
        """Test that gene swap distance stays within the range dictated by severity"""
        original_gene = CompositeGene([RotateCells() for _ in range(100)])

        for _ in range(500):
            severity = default_rng().random()
            mutagen = ShuffleGenes(severity)
            mutant_gene = mutagen.mutate_CompositeGene(original_gene)

            # Find the distance the moved gene, moved
            distance_moved = max(
                abs(original_gene.child_genes.index(g) - mutant_gene.child_genes.index(g))
                for g in original_gene.child_genes
            )

            # Max allowed distance = max(1, int(severity * num_genes))
            max_allowed = max(1, int(severity * len(original_gene.child_genes)))

            self.assertLessEqual(distance_moved, max_allowed)

            ratio = distance_moved / max_allowed if max_allowed > 0 else 0
            print(f"severity={severity:.4f}  distance={distance_moved}  max_allowed={max_allowed}  ratio={ratio:.4f}")


    def test_persist_reload(self):
        """Test that ShuffleGenes can be persisted and reloaded"""
        mutagen = ShuffleGenes(0.5, 0.022)
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
            self.assertEqual(reloaded.severity, 0.5)