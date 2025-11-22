import unittest
import uuid
from numpy.random import Generator, default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from typing import Set

import roxene
from ConnectNeurons_test import build_organism
from roxene.tic_tac_toe import Trial
from roxene.tic_tac_toe.outcome import Outcome
from roxene.tic_tac_toe.players import Player, ManualPlayer, REQUIRED_INPUTS, REQUIRED_OUTPUTS

SEED = 235869903

class Trial_test(unittest.TestCase):

    def test_occupied_square(self):
        p1 = Player()
        p2 = Player()
        trial = Trial(p1, p2)
        trial.run(queued_input=[(0, 0), (2, 2), (0, 1), (0, 0)])
        self.assertEqual(trial.moves[0].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[1].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[2].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[3].outcomes, {Outcome.INVALID_MOVE, Outcome.LOSE})

    def test_win(self):
        p1 = Player()
        p2 = Player()
        trial = Trial(p1, p2)
        trial.run(queued_input=[(1, 1), (0, 1), (0, 0), (1, 0), (2, 2)])
        self.assertEqual(trial.moves[0].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[1].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[2].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[3].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[4].outcomes, {Outcome.VALID_MOVE, Outcome.WIN})

    def test_simple_game_tie(self):
        p1 = Player()
        p2 = Player()
        trial = Trial(p1, p2)
        trial.run(queued_input=[(1, 1), (0, 0), (2, 2), (0, 2), (0, 1), (2, 1), (1, 0), (1, 2), (2, 0)])
        self.assertEqual(len(trial.moves), 9)
        self.assertTrue(all(move.outcomes=={Outcome.VALID_MOVE} for move in trial.moves))

    def test_save_trial_before_run(self):
        rng: Generator = default_rng(seed=SEED)
        uuid.uuid4 = lambda: uuid.UUID(bytes=rng.bytes(16))

        engine = create_engine("sqlite://")
        roxene.EntityBase.metadata.create_all(engine)

        org_1 = build_organism(rng=rng)
        org_2 = build_organism(rng=rng)
        trial = Trial(Player(org_1, 'X'), Player(org_2, 'O'))
        trial_id = trial.id

        with Session(engine) as session:
            session.add(org_1)
            session.add(org_2)
            session.add(trial)
            session.commit()

        with Session(engine) as session:
            trial_2: Trial = session.get(Trial, trial_id)
            self.assertIsNotNone(trial_2)
            self.assertTrue(len(trial_2.participants) == 2, "Participants should be populated")

            player_letters: Set[str] = set()
            for participant in trial_2.participants:
                self.assertIsNotNone(participant.organism)
                self.assertIsNotNone(participant.letter)
                player_letters.add(participant.letter)
            self.assertSetEqual(player_letters, {'X', 'O'})

            self.assertEqual(len(trial_2.moves), 0)
            self.assertIsNone(trial_2.start_date)
            self.assertIsNone(trial_2.end_date)
            self.assertEqual(trial_2.id, trial_id)

    def test_save_trial_before_and_after_run(self):
        rng: Generator = default_rng(seed=SEED)
        uuid.uuid4 = lambda: uuid.UUID(bytes=rng.bytes(16))

        engine = create_engine("sqlite://")
        roxene.EntityBase.metadata.create_all(engine)

        with Session(engine) as session:
            org_1 = build_organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS, rng=rng)
            org_2 = build_organism(input_names=REQUIRED_INPUTS, output_names=REQUIRED_OUTPUTS, rng=rng)
            trial = Trial(Player(org_1, 'X'), Player(org_2, 'O'))
            trial_id = trial.id

            session.add(trial)
            session.add(org_1)
            session.add(org_2)
            session.commit()

            for part in trial.participants:
                self.assertIsNotNone(part.organism)
                self.assertTrue("There should be inputs in the organism", len(part.organism.inputs) > 0)

        for part in trial.participants:
            self.assertIsNotNone(part.organism)
            self.assertTrue("There should be inputs in the organism", len(part.organism.inputs) > 0)

        with Session(engine) as session:
            trial_2 = session.get(Trial, trial_id)
            trial_2.run()
            session.commit()

        with Session(engine) as session:
            trial_3: Trial = session.get(Trial, trial_id)
            self.assertIsNotNone(trial_3)
            self.assertTrue(len(trial_3.participants) == 2, "Participants should be populated")

            player_letters: Set[str] = set()
            for participant in trial_3.participants:
                self.assertIsNotNone(participant.organism)
                self.assertIsNotNone(participant.letter)
                player_letters.add(participant.letter)
            self.assertSetEqual(player_letters, {'X', 'O'})

            self.assertGreater(len(trial_3.moves), 0)
            self.assertIsNotNone(trial_3.start_date)
            self.assertIsNotNone(trial_3.end_date)
