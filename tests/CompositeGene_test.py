import unittest
from unittest.mock import Mock, call

from roxene import Organism, Gene
from roxene.genes import CompositeGene


class CompositeGene_test(unittest.TestCase):

    def test_execute_once(self):
        organism: Organism = Mock(Organism)
        mock_genes = [Mock(Gene) for i in range(10)]

        gene: CompositeGene = CompositeGene(mock_genes)
        gene.execute(organism)
        for mock_gene in mock_genes:
            mock_gene.execute.assert_called_once_with(organism)

    def test_execute_twice(self):
        organism: Organism = Mock(Organism)
        mock_genes = [Mock(Gene) for i in range(10)]
        gene: CompositeGene = CompositeGene(mock_genes, 2)
        gene.execute(organism)
        for mock_gene in mock_genes:
            mock_gene.execute.assert_has_calls([call(organism), call(organism)])




