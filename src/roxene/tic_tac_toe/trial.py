import logging
import uuid
from datetime import datetime
from enum import Enum, auto
from sqlalchemy import ForeignKey, CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, Set
from typing import Tuple

from ..organism import Organism
from ..persistence import EntityBase

WIN_SETS = [
    {(0, 0), (0, 1), (0, 2)},
    {(1, 0), (1, 1), (1, 2)},
    {(2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 1), (1, 1), (2, 1)},
    {(0, 2), (1, 2), (2, 2)},
    {(0, 2), (1, 1), (2, 0)},
    {(0, 0), (1, 1), (2, 2)},
]


class Move:
    def __init__(self, letter: str, initial_board_state: List[List[str]]):
        self.letter = letter
        self.initial_board_state = [row.copy() for row in initial_board_state]
        self.position: tuple(int, int) = None
        self.outcomes: set[Outcome] = set()
        self.resultant_board_state: list[list[str]] = None


class Trial:
    __tablename__ = 'trial'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    participants: Mapped[Set["Player"]] = relationship("Player", back_populates='trial')
    start_date: Mapped[datetime] = mapped_column(nullable=False)
    end_date: Mapped[datetime]

    def __init__(self, player_1: "Player", player_2: "Player"):
        self.id = uuid.uuid4()
        self.board = [[None, None, None], [None, None, None], [None, None, None]]
        self.next_player_letter = 'X'
        self.moves: list[Move] = []
        self.players: map[str, "Player"] = {
            'X': player_1,
            'O': player_2
        }

    def is_finished(self) -> bool:
        return len(self.moves) == 9 or \
               any(filter(lambda move: Outcome.WIN in move.outcomes or Outcome.LOSE in move.outcomes, self.moves))

    def run(self):
        logging.info(f"Beginning trial for {[str(p) for p in self.players.values()]}")
        while not (self.is_finished()):
            current_player_letter = self.next_player_letter
            self.next_player_letter = 'X' if self.next_player_letter == 'O' else 'O'
            current_player: "Player" = self.players[current_player_letter]
            this_move = Move(letter=current_player_letter, initial_board_state=self.board)
            try:
                move_coords = current_player.get_move_coords(self.board)
                this_move.position = move_coords
                existing_square_value = self.board[move_coords[0]][move_coords[1]]
                if existing_square_value:
                    this_move.outcomes |= {Outcome.INVALID_MOVE, Outcome.LOSE}
                else:
                    this_move.outcomes |= {Outcome.VALID_MOVE}
                    self.board[move_coords[0]][move_coords[1]] = current_player_letter
                    this_move.resultant_board_state = [row.copy() for row in self.board]
                    win_sets_with_this_square = filter(lambda win_set: move_coords in win_set, WIN_SETS)
                    for win_set in win_sets_with_this_square:
                        square_values = [self.board[x][y] for x, y in win_set]
                        is_winning_move = all(square == current_player_letter for square in square_values)
                        if is_winning_move:
                            this_move.outcomes |= {Outcome.WIN}
            except TimeoutError:
                this_move.outcomes |= {Outcome.TIMEOUT, Outcome.LOSE}
            self.moves.append(this_move)

class Outcome(Enum):
    WIN = auto()
    LOSE = auto()
    TIE = auto()
    TIMEOUT = auto()
    VALID_MOVE = auto()
    INVALID_MOVE = auto()


LOW_THRESHOLD = -0.5
HIGH_THRESHOLD = 0.5
MAX_UPDATES = 1_000

INPUT_READY = "INPUT_READY"
OUTPUT_READY = "OUTPUT_READY"

REQUIRED_INPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [INPUT_READY]
REQUIRED_OUTPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [OUTPUT_READY]


class Player(EntityBase):
    __tablename__ = "trial_participant"

    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trial.id"), primary_key=True)
    trial: Mapped[Trial] = relationship(Trial, back_populates='participants')

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"), unique=True)
    organism: Mapped[Organism] = relationship(Organism)

    letter: Mapped[str] = mapped_column(CHAR(1), primary_key=True)

    def __init__(self, trial: Trial, organism: Organism, letter: str):
        self.trial = trial
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
        assert (len(digit_strings) == 2)
        return tuple(int(s) for s in digit_strings)
