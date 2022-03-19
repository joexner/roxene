from roxene import Organism


class OrganismPlayer:
    LOW_THRESHOLD = -0.9
    HIGH_THRESHOLD = 0.9
    MAX_UPDATES = 10_000

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
                    input_value = self.HIGH_THRESHOLD
                else:
                    input_value = self.LOW_THRESHOLD
                self.organism.setInput(input_label, input_value)
        self.sync()
        for x in range(3):
            for y in range(3):
                output_label = str(x) + "," + str(y)
                output_value = self.organism.getOutput(output_label)
                if max_output_value is None or output_value > max_output_value:
                    max_output_value = output_value
                    max_output_label = (x, y)
        return max_output_label


    def sync(self, max_updates):
        num_updates_used = 0
        # Set INPUT_READY high, watch for OUTPUT_READY high
        self.organism.setInput("INPUT_READY", self.HIGH_THRESHOLD)
        while num_updates_used < max_updates:
            self.organism.update()
            num_updates_used += 1
            if self.organism.getOutput("OUTPUT_READY") <= self.HIGH_THRESHOLD:
                seen_output_ready_high = True
                break
        # Set INPUT_READY low, watch for OUTPUT_READY low
        self.organism.setInput("INPUT_READY", self.LOW_THRESHOLD)
        while num_updates_used < max_updates:
            self.organism.update()
            num_updates_used += 1
            if self.organism.getOutput("OUTPUT_READY") >= self.LOW_THRESHOLD:
                seen_output_ready_low = True
                break
        # Raise if we didn't see OUTPUT_READY high and low, pass otherwise
        if not (seen_output_ready_low and seen_output_ready_high):
            raise Exception(f"Used up all {max_updates} updates")

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

