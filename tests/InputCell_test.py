import unittest
from numpy.random import Generator, default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import InputCell
from roxene.persistence import EntityBase


class InputCell_test(unittest.TestCase):

    def test_save_input_cell(self):
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        rng: Generator = default_rng()
        initial_value = rng.uniform(-1, 1)
        ic1 = InputCell(initial_value)
        icid = ic1.id

        with Session(engine) as session:
            session.add(ic1)
            session.commit()

        with Session(engine) as session:
            ic2 = session.get(InputCell, icid)
            self.assertFalse(ic2 is None)
            self.assertEqual(ic2.get_output(), initial_value)
            new_value = rng.uniform(-1, 1)
            ic2.set_output(new_value)
            session.commit()

        with Session(engine) as session:
            ic3 = session.get(InputCell, icid)
            self.assertFalse(ic3 is None)
            self.assertEqual(ic3.get_output(), new_value)
