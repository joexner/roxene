import unittest

from tic_tac_toe.players import ManualPlayer


class ManualPlayer_test(unittest.TestCase):
    def test_get_move_coords(self):
        board = [[None, None, None], [None, 'X', 'O'], [None, None, None]]
        player = ManualPlayer('X')
        move_coords = player.get_move_coords(board, "2 1")
        self.assertEqual((2,1), move_coords)


if __name__ == '__main__':
    unittest.main()
