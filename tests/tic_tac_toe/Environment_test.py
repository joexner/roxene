import logging
import unittest

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from roxene import Organism
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
        trial_2.run()

        environment.complete_trial(trial_1)

        with Session(engine) as session:
            session.add_all([trial_1, trial_2])
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




if __name__ == '__main__':
    unittest.main()
