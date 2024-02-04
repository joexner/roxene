import tensorflow as tf
from sqlalchemy import create_engine

from roxene import Organism
from roxene.persistence import EntityBase
from roxene.tic_tac_toe.players import REQUIRED_INPUTS, REQUIRED_OUTPUTS
from roxene.tic_tac_toe.population import Population


# from organism import Organism


class Population_test(tf.test.TestCase):

    def test_add_and_sample_non_idle(self):
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        pop: Population = Population(engine, [])
        num_orgs = 52
        for n in range(num_orgs):
            organism = Organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS)
            pop.add(organism)
        orgs = pop.sample(num_organisms=num_orgs)
        self.assertEqual(num_orgs, len(orgs))
