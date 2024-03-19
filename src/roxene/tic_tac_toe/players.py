from __future__ import annotations

import logging
import uuid
from sqlalchemy import ForeignKey, CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Tuple

# import roxene.tic_tac_toe as ttt
from roxene import EntityBase, Organism

# from .trial import Trial

LOW_THRESHOLD = -0.5
HIGH_THRESHOLD = 0.5
MAX_UPDATES = 1_000

INPUT_READY = "INPUT_READY"
OUTPUT_READY = "OUTPUT_READY"

REQUIRED_INPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [INPUT_READY]
REQUIRED_OUTPUTS = [str(x) + ',' + str(y) for x in range(3) for y in range(3)] + [OUTPUT_READY]


class Player(EntityBase):
    __tablename__ = "player"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trial.id"))
    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"))

    trial: Mapped[Trial] = relationship("Trial", back_populates='participants', lazy="joined")
    organism: Mapped[Organism] = relationship(Organism, lazy="joined")

    letter: Mapped[str] = mapped_column(CHAR(1), primary_key=True)

    def __init__(self, organism: Organism, letter: str = None):
        self.id = uuid.uuid4()
        self.organism = organism
        self.letter = letter

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
        logger = logging.getLogger(str(self.organism)).getChild("player")
        logger.info("Beginning sync")
        num_updates_used = 0
        next_log_at_update_number = 10

        # Set INPUT_READY high, watch for OUTPUT_READY high
        logger.debug("Waiting for high output")
        seen_output_ready_high = False
        self.organism.set_input(INPUT_READY, HIGH_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                logger.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value >= HIGH_THRESHOLD:
                seen_output_ready_high = True
                logger.debug("Got high output")
                break
        if not seen_output_ready_high:
            logger.info(f"Failed to sync in {max_updates} updates")
            raise TimeoutError(f"Used up all {max_updates} updates")

        # Set INPUT_READY low, watch for OUTPUT_READY low
        logger.debug("Waiting for low output")
        seen_output_ready_low = False
        self.organism.set_input(INPUT_READY, LOW_THRESHOLD)
        while num_updates_used < max_updates:
            if num_updates_used == next_log_at_update_number:
                next_log_at_update_number *= 2
                logger.debug(f"Beginning update {num_updates_used}")
            self.organism.update()
            num_updates_used += 1
            output_value = self.organism.get_output(OUTPUT_READY)
            if output_value <= LOW_THRESHOLD:
                seen_output_ready_low = True
                logger.debug("Got low output")
                break

        # Raise if we didn't see OUTPUT_READY high and low, pass otherwise
        if not seen_output_ready_low:
            logger.info(f"Failed to sync in {max_updates} updates")
            raise TimeoutError(f"Used up all {max_updates} updates")

        logger.info(f"{self} synced in {num_updates_used} updates")


class ManualPlayer(Player):
    def __init__(self, queued_input=[]):
        self.id = uuid.uuid4()
        self.queued_input = list(queued_input)

    def get_move_coords(self, board) -> Tuple[int]:
        if self.queued_input:
            return self.queued_input.pop(0)
        else:
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
            user_input = input(f"Next move for {self.letter}:")
        digit_strings = user_input.split(sep=" ")
        assert (len(digit_strings) == 2)
        return tuple(int(s) for s in digit_strings)
