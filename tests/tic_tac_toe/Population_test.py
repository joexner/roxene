import logging
import unittest

from numpy.random import default_rng, Generator
from sqlalchemy.orm import sessionmaker

from roxene import Organism
from roxene.tic_tac_toe import Population, Environment
from util import get_engine

SEED = 1123581321

class Population_test(unittest.TestCase):

    rng: Generator = default_rng(SEED)

    def test_add_and_sample_non_idle(self):
        engine = get_engine()
        pop: Population = Population()
        num_orgs = 100

        # Add a bunch of empty (no cells) organisms to the population
        for n in range(num_orgs):
            organism = Organism()
            with sessionmaker(engine).begin() as session:
                pop.add(organism, session)

        with sessionmaker(engine).begin() as session:
            orgs = pop.sample(num_orgs, False, self.rng, session)

        self.assertEqual(num_orgs, len(orgs))


    def test_sample_idle(self):
        # Build an environment for creating organisms and starting trials
        engine = get_engine()
        env = Environment(SEED, engine)

        # Get direct access to the population
        pop = env.population

        num_orgs = 5  # Create 5 organisms
        logging.info(f"Populating {num_orgs} organisms")
        env.populate(num_orgs, {"input_size": 2, "feedback_size": 2, "hidden_size": 2})

        logging.info(f"Done populating organisms")

        # Start 2 trials, which will use 4 organisms (2 per trial)
        logging.info("Starting first trial")
        env.start_trial()
        logging.info("Starting second trial")
        env.start_trial()

        # Should have 1 idle organism left

        # Sample 1 idle organism - this should work
        with sessionmaker(engine).begin() as session:
            idle_orgs = pop.sample(1, True, self.rng, session)

        self.assertEqual(1, len(idle_orgs))
        logging.info(f"Successfully sampled 1 idle organism")

        # Try to sample 2 idle organisms - this should fail since there's only 1
        try:
            with sessionmaker(engine).begin() as session:
                pop.sample(2, True, self.rng, session)
            self.fail("Should not be able to sample 2 idle orgs when only 1 exists")
        except BaseException as expected_ex:
            logging.info(expected_ex)
