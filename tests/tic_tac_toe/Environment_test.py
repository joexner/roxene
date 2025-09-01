import logging
import unittest

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from roxene import Organism
from roxene.mutagens import CreateNeuronMutagen, CNLayer
from roxene.tic_tac_toe import Trial, Player, Environment, Outcome
from util import get_engine

SEED = 11235


class Environment_test(unittest.TestCase):

    logger: logging.Logger = logging.getLogger(__name__)

    def test_get_relevant_moves(self):
        engine: Engine = get_engine()

        environment: Environment = Environment(123, engine)

        o1, o2, o3 = Organism(), Organism(), Organism()
        o1_id, o2_id, o3_id = o1.id, o2.id, o3.id

        trial_1 = Trial(Player(o1, queued_input=[(0, 0), (0, 1), (0, 2)]),
                        Player(o2, queued_input=[(1, 0), (1, 1)]))

        trial_2 = Trial(Player(o1, queued_input=[(0, 0), (0, 1)]),
                        Player(o3, queued_input=[(1, 0), (0, 0)]))

        trial_1.run()
        environment.complete_trial(trial_1)

        trial_2.run()
        environment.complete_trial(trial_2)

        with Session(engine) as session:
            self.assertEqual(5, len(list(environment.get_relevant_moves([o1_id], session))))
            self.assertEqual(4, len(list(environment.get_relevant_moves([o2_id, o3_id], session))))


    def test_start_and_run_trial(self):
        # Build an environment with a population with 2 organisms, start a trial
        env = Environment(SEED, get_engine())
        env.populate(2)
        trial = env.start_trial()

        trial.run(timeout=100)

        self.logger.info(f"Trial {trial.id} done, {len(trial.moves)} moves")

        # Save the trial to the database
        env.complete_trial(trial)

        moves: list = list(trial.moves)
        self.assertGreater(len(moves), 0)
        last_move = moves.pop()
        outcomes = last_move.outcomes
        ended = Outcome.WIN in outcomes or Outcome.LOSE in outcomes
        self.assertTrue(ended, "Trial should have ended in a win or loss")


    def test_cull(self):
        # Build an environment with a larger population
        env = Environment(SEED, get_engine())
        initial_population_size = 50
        env.populate(initial_population_size)

        # Count organisms before culling
        with env.sessionmaker() as session:
            count_before = env.population.count(session)

        self.assertEqual(initial_population_size, count_before)

        # Cull some organisms
        num_to_cull = 3
        env.cull(num_to_cull)

        # Count organisms after culling
        with env.sessionmaker() as session:
            count_after = env.population.count(session)

        self.assertEqual(count_before - num_to_cull, count_after)

        count_before = count_after

        # Cull some more organisms
        num_to_cull = 31
        env.cull(num_to_cull)

        # Count organisms after culling
        with env.sessionmaker() as session:
            count_after = env.population.count(session)

        self.assertEqual(count_before - num_to_cull, count_after)

    def test_breed(self):
        env = Environment(SEED, get_engine())

        initial_population_size = 100
        env.populate(initial_population_size)

        # Count organisms before breeding, because
        with env.sessionmaker() as session:
            count_before = env.population.count(session)

        self.assertEqual(initial_population_size, count_before)

        # Breed a few, make sure they show up
        num_to_breed = 3
        env.breed(num_to_breed)

        with env.sessionmaker() as session:
            count_after = env.population.count(session)

        # Breeding should increase the population
        self.assertEqual(count_after, count_before + num_to_breed)

        count_before = count_after

        # Breed a few more, make sure they show up too
        num_to_breed = 19
        env.breed(num_to_breed)

        with env.sessionmaker() as session:
            count_after = env.population.count(session)

        # Breeding should increase the population
        self.assertEqual(count_after, count_before + num_to_breed)

    def test_clone(self):
        # Test cloning functionality by checking that a clone is created with and without mutations
        env = Environment(SEED, get_engine())

        # Add a mutagen to ensure mutation occurs during cloning
        env.mutagens = [CreateNeuronMutagen(CNLayer.input_initial_value, base_susceptibility=1.0)]

        # Create a single organism to clone
        env.populate(1)

        with env.sessionmaker() as session:
            # Get the only organism
            original_organism = session.scalar(select(Organism))
            self.assertIsNotNone(original_organism)
            original_id = original_organism.id
            original_genotype = original_organism.genotype

            # Test clone with mutation (default behavior)
            clone = env.clone(original_id, session)

        # Verify the clone is a new organism (different ID)
        self.assertIsNot(clone, original_organism)
        self.assertNotEqual(clone.id, original_id)

        # Verify the clone is functional and has a different genotype (due to mutation)
        self.assertIsNotNone(clone.genotype)
        self.assertIsNot(clone.genotype, original_genotype)
        self.assertGreaterEqual(len(clone.cells), 0)


if __name__ == '__main__':
    unittest.main()
