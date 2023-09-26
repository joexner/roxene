import unittest
import tensorflow as tf
from sqlalchemy import create_engine

# from organism import Organism
from persistence import EntityBase
from tic_tac_toe.players import REQUIRED_OUTPUTS, REQUIRED_INPUTS
from tic_tac_toe.population import Population


# class Population_test(tf.TestCase):
#
#     def test_everything(self):
#         engine = create_engine("sqlite://")
#         EntityBase.metadata.create_all(engine)
#
#         pop: Population = Population(engine, [])
#         pop.add(Organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS))
#         self.assertEqual(True, False)  # add assertion here
#
#
# if __name__ == '__main__':
#     unittest.main()
