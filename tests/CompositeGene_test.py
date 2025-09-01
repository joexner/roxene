import unittest
from unittest.mock import Mock, call

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import Gene, Organism, EntityBase
from roxene.genes import CompositeGene, RotateCells


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

    def test_persistence(self):

        a = RotateCells()
        b = RotateCells(RotateCells.Direction.FORWARD)
        c = RotateCells()

        cg: CompositeGene = CompositeGene([a, b, c], 4)

        cg_id = cg.id
        original_cg_child_ids = [g.id for g in cg.child_genes]
        original_cg_iterations = cg.iterations

        cg2 = CompositeGene([c, a], 5)

        cg2_id = cg2.id
        original_cg2_child_ids = [g.id for g in cg2.child_genes]
        original_cg2_iterations = cg2.iterations

        engine = create_engine("sqlite://")

        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            session.add_all([cg, cg2, a, b, c])
            session.commit()

        with Session(engine) as session:
            reloaded = session.get(CompositeGene, cg_id)
            reloaded_gene_ids = [gene.id for gene in reloaded.child_genes]
            self.assertEqual(reloaded_gene_ids, original_cg_child_ids)
            self.assertEqual(reloaded.iterations, original_cg_iterations)

            reloaded_2 = session.get(CompositeGene, cg2_id)
            reloaded_2_gene_ids = [gene.id for gene in reloaded_2.child_genes]
            self.assertEqual(reloaded_2_gene_ids, original_cg2_child_ids)
            self.assertEqual(reloaded_2.iterations, original_cg2_iterations)
