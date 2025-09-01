import unittest
from parameterized import parameterized
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ConnectNeurons_test import build_organism
from roxene import EntityBase
from roxene.genes import RotateCells


class RotateCells_test(unittest.TestCase):

    @parameterized.expand([
        (RotateCells.Direction.FORWARD),
        (RotateCells.Direction.BACKWARD),
    ])
    def test_execute(self, direction):
        num_neurons = 5
        organism = build_organism(num_neurons, neuron_input_size=1)
        cells_before_execution = list(organism.cells)
        gene = RotateCells(direction)
        gene.execute(organism)
        cells_after_execution =  list(organism.cells)
        cutoff = (-1 * direction) % num_neurons
        expected_cells_after_execution = cells_before_execution[cutoff:] + cells_before_execution[:cutoff]
        self.assertEqual(cells_after_execution, expected_cells_after_execution)

    def test_persistence(self):
        forward = RotateCells(RotateCells.Direction.FORWARD)
        forward_id = forward.id

        backward = RotateCells(RotateCells.Direction.BACKWARD)
        backward_id = backward.id

        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            session.add_all([forward, backward])
            session.commit()

        with Session(engine) as session:
            reloaded_forward = session.get(RotateCells, forward_id)
            self.assertEqual(reloaded_forward.direction, RotateCells.Direction.FORWARD)

            reloaded_backward = session.get(RotateCells, backward_id)
            self.assertEqual(reloaded_backward.direction, RotateCells.Direction.BACKWARD)
