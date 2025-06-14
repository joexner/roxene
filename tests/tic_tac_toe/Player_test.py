import random
import unittest
from unittest.mock import Mock, call

from roxene import Organism
from roxene.tic_tac_toe import Player

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


class Player_test(unittest.TestCase):

    def test_get_move_coords(self):

        organism: Organism = Mock(Organism)
        player = Player(organism, 'X')
        board = generateBoard(numToFill=6)

        # Let the mock Organism's outputs vary continuously, randomly between -1 and 1
        # This will eventually convince the Player that we're ready wih our other output
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

    def test_sync_1_high_1_low(self):
        """Let the organism show high on "OUTPUT_READY", then low to satisfy the Player that it's ready"""
        organism = Mock(Organism)
        organism.get_output.side_effect = [0.5, -0.5]
        player = Player(organism=organism, letter='X')
        player.sync(timeout=10)

        # The player showed the Organism 2 set_input calls, to "INPUT_READY", high then low
        self.assertEqual(organism.set_input.call_count, 2)
        organism.set_input.assert_has_calls([call("INPUT_READY", 0.5), call("INPUT_READY", -0.5)])

        self.assertEqual(organism.get_output.call_count, 2)

    def test_sync_10_low(self):
        """Let the organism show only low on "OUTPUT_READY", until it triggers a timeout"""
        organism = Mock(Organism)
        max_updates = 100

        organism.get_output.side_effect = [-0.5] * 10
        player = Player(organism=organism, letter='X')
        try:
            player.sync(timeout=10)
            self.fail("Should have failed")
        except TimeoutError as expected:
            pass

        # Player only showed the Organism INPUT_READY high, once
        organism.set_input.assert_has_calls([call("INPUT_READY", 0.5)])

        # The Player asked for the Organism's output 10 times before timing out
        self.assertEqual(organism.get_output.call_count, 10)

    def test_sync_10_high(self):
        """Let the organism show only high on "OUTPUT_READY", until it triggers a timeout"""
        organism = Mock(Organism)
        max_updates = 100

        organism.get_output.side_effect = [0.5] * 10
        player = Player(organism=organism, letter='X')
        try:
            player.sync(timeout=10)
            self.fail("Should have failed")
        except TimeoutError as expected:
            pass

        # Player only showed the Organism INPUT_READY high, once
        organism.set_input.assert_has_calls([call("INPUT_READY", 0.5), call("INPUT_READY", -0.5)])

        # The Player asked for the Organism's output 10 times before timing out
        self.assertEqual(organism.get_output.call_count, 10)
