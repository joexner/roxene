import unittest
from random import Random

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import Organism, EntityBase
from roxene.cells import InputCell
from roxene.genes import CreateInputCell

SEED = 84592184


class CreateInputCell_test(unittest.TestCase):

    def test_execute(self):
        rng = Random(SEED)
        org = Organism()
        initial_value = rng.uniform(-1, 1)
        gene = CreateInputCell(initial_value)
        self.assertEqual(0, len(org.cells))
        gene.execute(org)
        self.assertEqual(len(org.cells), 1)
        new_cell: InputCell = org.cells[0]
        self.assertEqual(new_cell.get_output(), initial_value)

    def test_persistence(self):
        initial_value = -0.1582
        gene = CreateInputCell(initial_value)
        gene_id = gene.id

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            session.add(gene)
            session.commit()

        with Session(engine) as session:
            reloaded = session.get(CreateInputCell, gene_id)
            self.assertEqual(reloaded.initial_value, initial_value)
