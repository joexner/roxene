import unittest
from unittest.mock import Mock

from roxene.tic_tac_toe import ManualPlayer, Trial
from roxene.tic_tac_toe.trial import Outcome


class Trial_test(unittest.TestCase):

    def test_occupied_square(self):
        p1 = Mock(ManualPlayer)
        p2 = Mock(ManualPlayer)
        p1.get_move_coords.side_effect=[(0,0), (0,1)]
        p2.get_move_coords.side_effect=[(2,2), (0,0)]
        trial = Trial(p1, p2)
        trial.run()
        self.assertEqual(trial.moves[0].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[1].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[2].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[3].outcomes, {Outcome.INVALID_MOVE, Outcome.LOSE})

    def test_win(self):
        p1 = Mock(ManualPlayer)
        p2 = Mock(ManualPlayer)
        p1.get_move_coords.side_effect=[(1,1), (0,0), (2,2)]
        p2.get_move_coords.side_effect=[(0,1), (1,0)]
        trial = Trial(p1, p2)
        trial.run()
        self.assertEqual(trial.moves[0].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[1].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[2].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[3].outcomes, {Outcome.VALID_MOVE})
        self.assertEqual(trial.moves[4].outcomes, {Outcome.VALID_MOVE, Outcome.WIN})

    def test_simple_game_tie(self):
        p1 = Mock(ManualPlayer)
        p2 = Mock(ManualPlayer)
        p1.get_move_coords.side_effect=[(1,1), (2,2), (0,1), (1,0), (2,0)]
        p2.get_move_coords.side_effect=[(0,0), (0,2), (2,1), (1,2)]
        trial = Trial(p1, p2)
        trial.run()
        self.assertEqual(len(trial.moves), 9)
        self.assertTrue(all(move.outcomes=={Outcome.VALID_MOVE} for move in trial.moves))
