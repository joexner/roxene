import logging
from abc import ABC, abstractmethod
from typing import Tuple

from roxene import Organism

LOW_THRESHOLD = -0.5
HIGH_THRESHOLD = 0.5
MAX_UPDATES = 1_000

INPUT_READY = "INPUT_READY"
OUTPUT_READY = "OUTPUT_READY"

REQUIRED_INPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [INPUT_READY]
REQUIRED_OUTPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [OUTPUT_READY]


class Player(ABC):
    @abstractmethod
    def get_move_coords(self, board) -> Tuple[int]:
        pass


class OrganismPlayer(Player):

    def __init__(self, organism: Organism, letter: str):
        self.organism = organism
        self.letter = letter
        self.logger = logging.getLogger(str(organism)).getChild("player")

    def get_move_coords(self, board) -> Tuple[int]:
        for x in range(3):
            for y in range(3):
                input_label = str(x) + "," + str(y)
                if board[x][y] is None:
                    input_value = 0
                elif board[x][y] == self.letter:
                    input_value = HIGH_THRESHOLD
                else:
                    input_value = LOW_THRESHOLD
                self.organism.set_input(input_label, input_value)
        self.sync(MAX_UPDATES)
        max_output_value = None
        for x in range(3):
            for y in range(3):
                output_label = str(x) + "," + str(y)
                output_value = self.organism.get_output(output_label)
                if max_output_value is None or output_value > max_output_value:
                    max_output_value = output_value
                    max_output_label = (x, y)
        return max_output_label

    def __str__(self):
        return f"{str(self.organism)}.player"

    def sync(self, max_updates):
        self.logger.info("Beginning sync")
        num_updates_used = 0
        next_log_at_update_number = 10

        # Set INPUT_READY high, watch for OUTPUT_READY high
        self.logger.debug("Waiting for high output")
        seen_output_ready_high = False
        self.organism.set_input(INPUT_READY, HIGH_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                self.logger.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value >= HIGH_THRESHOLD:
                seen_output_ready_high = True
                self.logger.debug("Got high output")
                break
        if not seen_output_ready_high:
            self.logger.info(f"Failed to sync in {max_updates} updates")
            raise TimeoutError(f"Used up all {max_updates} updates")

        # Set INPUT_READY low, watch for OUTPUT_READY low
        self.logger.debug("Waiting for low output")
        seen_output_ready_low = False
        self.organism.set_input(INPUT_READY, LOW_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                self.logger.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value <= LOW_THRESHOLD:
                seen_output_ready_low = True
                self.logger.debug("Got low output")
                break

        # Raise if we didn't see OUTPUT_READY high and low, pass otherwise
        if not seen_output_ready_low:
            self.logger.info(f"Failed to sync in {max_updates} updates")
            raise TimeoutError(f"Used up all {max_updates} updates")

        self.logger.info(f"{self} synced in {num_updates_used} updates")


class ManualPlayer(Player):
    def __init__(self, letter: str):
        self.letter = letter

    def get_move_coords(self, board, raw_input=None) -> Tuple[int]:
        print()
        for x in range(5):
            for y in range(5):
                if x % 2 == 0 and y % 2 == 0:
                    value = board[int(x / 2)][int(y / 2)] or ' '
                    print(value, end='')
                elif x % 2 == 1 and y % 2 == 0:
                    print("-", end='')
                elif x % 2 == 1 and y % 2 == 1:
                    print("+", end='')
                elif x % 2 == 0 and y % 2 == 1:
                    print("|", end='')
            print()
        user_input = raw_input or input(f"Next move for {self.letter}:")
        digit_strings = user_input.split(sep=" ")
        assert(len(digit_strings) == 2)
        return tuple(int(s) for s in digit_strings)

