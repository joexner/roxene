import random
import unittest
from unittest.mock import Mock, call

from roxene import Organism
from roxene.tic_tac_toe import OrganismPlayer

MAX_VALUE = 0.5
MIN_VALUE = -0.5


def generateBoard(numToFill):
    points = list()
    for x in range(3):
        for y in range(3):
            points.append((x, y))
    random.shuffle(points)  # Yeah yeah, too lazy to figure out how to seed the RNG...
    board = [[None] * 3 for _ in range(3)]
    for n in range(numToFill):
        point = points.pop()
        board[point[0]][point[1]] = 'X' if n % 2 == 1 else 'O'
    return board


class OrganismPlayer_test(unittest.TestCase):

    def test_get_move_coords(self):
        organism = Mock(Organism)
        player = OrganismPlayer(organism, letter='X')
        board = generateBoard(numToFill=6)

        # Let the mock Organism's outputs vary continuously, randomly between -1 and 1
        # This will eventually convince the OrganismPlayer that we're ready wih our other output
        organism.get_output.side_effect = lambda _: random.uniform(-1, 1)

        player.get_move_coords(board)

        # Check that the player set the inputs on the Organism correctly
        expected_set_input_calls = []
        for x in range(3):
            for y in range(3):
                value = 0
                if board[x][y]:
                    value = MAX_VALUE if board[x][y] == player.letter else MIN_VALUE
                expected_set_input_calls.append(call(f"{x},{y}", value))
        organism.set_input.assert_has_calls(expected_set_input_calls, any_order=True)

    def test_sync(self):

        # Let the organism show low on "OUTPUT_READY", then high
        # to satisfy the OrganismPlayer that it's ready
        organism = Mock(Organism)
        organism.get_output.side_effect = [0.5, -0.5]
        player = OrganismPlayer(organism=organism, letter='X')
        player.sync(max_updates=10)

        # The player showed the Organism 2 set_input calls, to "INPUT_READY", high then low
        self.assertEqual(organism.set_input.call_count, 2)
        organism.set_input.assert_has_calls([call("INPUT_READY", 0.5), call("INPUT_READY", -0.5)])

        self.assertEqual(organism.get_output.call_count, 2)
