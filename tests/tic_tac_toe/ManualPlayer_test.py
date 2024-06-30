import unittest

from roxene.tic_tac_toe import ManualPlayer, Player


class ManualPlayer_test(unittest.TestCase):

    def test_get_move_coords(self):
        board = [[None, None, None], [None, 'X', 'O'], [None, None, None]]
        player = Player(queued_input=[(2, 1), (0, 2)])
        move_coords = player.get_move_coords(board)
        self.assertEqual((2,1), move_coords)
        move_coords = player.get_move_coords(board)
        self.assertEqual((0, 2), move_coords)
