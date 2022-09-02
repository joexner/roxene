import logging

from roxene import Organism

LOW_THRESHOLD = -0.5
HIGH_THRESHOLD = 0.5
MAX_UPDATES = 1_000

INPUT_READY = "INPUT_READY"
OUTPUT_READY = "OUTPUT_READY"

REQUIRED_INPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [INPUT_READY]
REQUIRED_OUTPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [OUTPUT_READY]


class OrganismPlayer:

    def __init__(self, organism: Organism, letter: str):
        self.organism = organism
        self.letter = letter

    def get_move_coords(self, board) -> tuple[int]:
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


    def sync(self, max_updates):
        logging.info("Beginning sync")
        num_updates_used = 0
        next_log_at_update_number = 10

        # Set INPUT_READY high, watch for OUTPUT_READY high
        logging.debug(f"Beginning update {self.organism}")
        seen_output_ready_high = False
        self.organism.set_input(INPUT_READY, HIGH_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                logging.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value >= HIGH_THRESHOLD:
                seen_output_ready_high = True
                break

        # Set INPUT_READY low, watch for OUTPUT_READY low
        seen_output_ready_low = False
        self.organism.set_input(INPUT_READY, LOW_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                logging.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value <= LOW_THRESHOLD:
                seen_output_ready_low = True
                break

        # Raise if we didn't see OUTPUT_READY high and low, pass otherwise
        if not (seen_output_ready_high and seen_output_ready_low):
            raise TimeoutError(f"Used up all {max_updates} updates")
        else:
            logging.info(f"Organism synced in {num_updates_used} updates")

class ManualPlayer:
    def __init__(self, letter: str):
        self.letter = letter

    def get_move_coords(self, board, raw_input=None) -> tuple[int]:
        print()
        for x in range(5):
            for y in range(5):
                if x % 2 == 0 and y % 2 == 0:
                    value = board[int(x/2)][int(y/2)] or ' '
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

