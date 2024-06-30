import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ConnectNeurons_test import build_organism
from roxene import Organism
from roxene.persistence import EntityBase
from roxene.tic_tac_toe import Population
from roxene.tic_tac_toe.players import REQUIRED_INPUTS, REQUIRED_OUTPUTS

from numpy.random import default_rng

# from organism import Organism


SEED = 1123581321

class Population_test(unittest.TestCase):


    def test_add_and_sample_non_idle(self):
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            pop: Population = Population()
            num_orgs = 52
            for n in range(num_orgs):
                organism = Organism()
                pop.add(organism, session)
            orgs = pop.sample(num_orgs, False, session)
            self.assertEqual(num_orgs, len(orgs))

    def test_start_and_run_trial(self):


        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        for seed in range(11235, 12345):

            print(f"Trying seed {seed}")
            rng = default_rng(SEED)

            with Session(engine) as session:
                # Build a population, start a trial, verify results
                pop: Population = Population()

                num_orgs = 10
                for n in range(num_orgs):
                    organism = build_organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS, rng=rng)
                    pop.add(organism, session)

                trial = pop.start_trial(session)
                trial.run(timeout=100)
                pop.complete_trial(trial, session)

                moves = list(trial.moves)

                print(f"Trial {trial.id} from seed {seed} had {len(moves)} moves")

                if len(moves) > 1:
                    for move in moves:
                        print(move)
                    break
                else:
                    print(f"Trial {trial.id} from seed {seed} had {len(moves)} moves")

    def test_sample_idle(self):
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)

        with Session(engine) as session:

            # Build a population, make sure we can put them all in trials simultaneously
            pop: Population = Population()

            num_orgs = 20

            for n in range(num_orgs):
                organism = Organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS)
                pop.add(organism, session)

            for n in range(num_orgs // 2):
                pop.start_trial(session)


            try:
                pop.start_trial(session)
                self.fail("Should be out of idle orgs")
            except BaseException as expected_ex:
                print(expected_ex)
