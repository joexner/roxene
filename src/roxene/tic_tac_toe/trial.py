import logging
import uuid
from datetime import datetime
from enum import Enum, auto
from sqlalchemy import ForeignKey, CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List, Set

from organism import Organism
from tic_tac_toe.players import Player

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


class Participant:
    __tablename__ = "trial_participant"

    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trial.id"), primary_key=True)
    trial: Mapped['Trial'] = relationship('Trial', back_populates='participants')

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id"), unique=True)
    organism: Mapped[Organism] = relationship(Organism)

    letter: Mapped[str] = mapped_column(CHAR(1), primary_key=True)

    def __init__(self, trial: 'Trial', organism: Organism, letter: str):
        self.trial = trial
        self.organism = organism
        self.letter = letter

    def __str__(self):
        return self.name

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
    participants: Mapped[Set[Participant]] = relationship(Participant, back_populates='trial')
    start_date: Mapped[datetime] = mapped_column(nullable=False)
    end_date: Mapped[datetime]

    def __init__(self, player_1: Player, player_2: Player):
        self.id = uuid.uuid4()
        self.board = [[None, None, None], [None, None, None], [None, None, None]]
        self.next_player_letter = 'X'
        self.moves: list[Move] = []
        self.players: map[str, Player] = {
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
            current_player: Player = self.players[current_player_letter]
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
