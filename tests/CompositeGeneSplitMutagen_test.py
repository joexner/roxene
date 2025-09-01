import logging
import unittest
from typing import List

import numpy as np

from roxene import Gene
from roxene.genes import CompositeGene, RotateCells
from roxene.mutagens import CompositeGeneSplitMutagen

SEED = 11235

logger = logging.getLogger(__name__)

class CompositeGeneSplitMutagen_test(unittest.TestCase):

    def test_split_basic(self):
        # Parent CG with 10 iterations
        child_genes: List[Gene] = [RotateCells(RotateCells.Direction.BACKWARD),
                                   RotateCells(RotateCells.Direction.FORWARD),
                                   RotateCells(RotateCells.Direction.FORWARD)]
        original_gene = CompositeGene(child_genes=child_genes, iterations=10)

        # SUT, with 1.0 sus and 0 log-wiggle it will always apply #TODO verify this 👈
        mutagen = CompositeGeneSplitMutagen(1,0)

        rng = np.random.default_rng(SEED)
        for _ in range(20):
            mutant_gene = mutagen.mutate(original_gene, rng)

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
            logger.info(f"Split into {first.iterations} and {second.iterations}")

