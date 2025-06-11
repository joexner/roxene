import logging
import unittest

from numpy.random import default_rng, Generator
from sqlalchemy.orm import sessionmaker

from roxene import Organism
from roxene.tic_tac_toe import Population, Player, Trial
from util import get_engine

SEED = 1123581321

class Population_test(unittest.TestCase):


    def test_add_and_sample_non_idle(self):
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        pop: Population = Population()
        num_orgs = 100

        # Add a bunch of empty (no cells) organisms to the population
        for n in range(num_orgs):
            organism = Organism()
            with seshmaker.begin() as session:
                pop.add(organism, session)

        with seshmaker.begin() as session:
            orgs = pop.sample(num_orgs, False, rng, session)

        self.assertEqual(num_orgs, len(orgs))


    def test_sample_idle(self):
        rng: Generator = default_rng(SEED)
        engine = get_engine()
        seshmaker = sessionmaker(engine)
        with seshmaker() as session:
            pop = Population()
            num_orgs = 5  # Create 5 organisms
            logging.info(f"Populating {num_orgs} organisms")
            for _ in range(num_orgs):
                org = Organism()
                pop.add(org, session)
            session.commit()
            logging.info(f"Done populating organisms")

            # Start 2 trials, which will use 4 organisms (2 per trial)
            for _ in range(2):
                logging.info("Starting trial")
                orgs = pop.sample(2, True, rng, session)
                player_1 = Player(organism=orgs[0], letter='X')
                player_2 = Player(organism=orgs[1], letter='O')
                trial = Trial(player_1, player_2)
                session.add(trial)
                session.commit()

            # Should have 1 idle organism left

            # Sample 1 idle organism - this should work
            pop.sample(1, True, rng, session)
            logging.info(f"Successfully sampled 1 idle organism")

            # Try to sample 2 idle organisms - this should fail since there's only 1
            try:
                pop.sample(2, True, rng, session)
                self.fail("Should not be able to sample 2 idle orgs when only 1 exists")
            except BaseException as expected_ex:
                logging.info(expected_ex)
