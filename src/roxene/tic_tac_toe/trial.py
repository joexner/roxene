from __future__ import annotations

import itertools
import logging
import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Iterator, List

from .move import Move
from .outcome import Outcome
from .persistence import Point
from .players import Player
from ..persistence import EntityBase

# import roxene.tic_tac_toe as ttt

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


class Trial(EntityBase):
    __tablename__ = 'trial'
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    start_date: Mapped[datetime] = mapped_column(nullable=True)
    end_date: Mapped[datetime] = mapped_column(nullable=True)

    participants: Mapped[List[Player]] = relationship(Player, back_populates='trial', lazy="immediate")
    moves: Mapped[List[Move]] = relationship(Move, back_populates='trial', lazy="joined")

    def __init__(self, player_1: Player, player_2: Player):
        self.id = uuid.uuid4()
        self.participants = [player_1, player_2]
        self.moves = []  # Initialize the moves list
        player_1.letter = 'X'
        player_2.letter = 'O'

    def is_finished(self) -> bool:
        return len(self.moves) == 9 or \
               any(filter(lambda move: Outcome.WIN in move.outcomes or Outcome.LOSE in move.outcomes, self.moves))

    def run(self, timeout=1000, queued_input: List[tuple[int, int]] = None):
        self.start_date = datetime.now()
        board = [[None, None, None], [None, None, None], [None, None, None]]
        logging.info(f"Beginning trial for {[str(p) for p in self.participants]}")
        player_iter: Iterator[Player] = itertools.cycle(self.participants)
        # Put the X Player at the front
        for p in player_iter:
            if p.letter == 'O':
                break
        while not (self.is_finished()):
            current_player: Player = player_iter.__next__()
            this_move: Move = Move(player=current_player, initial_board_state=board)
            if current_player.organism is not None:
                this_move.organism = current_player.organism
            try:
                if queued_input is not None:
                    move_coords = queued_input.pop(0)
                else:
                    move_coords = current_player.get_move_coords(board, timeout)
                this_move.position = Point(move_coords[0], move_coords[1])
                existing_square_value = board[move_coords[0]][move_coords[1]]
                if existing_square_value:
                    this_move.outcomes |= {Outcome.INVALID_MOVE, Outcome.LOSE}
                else:
                    this_move.outcomes |= {Outcome.VALID_MOVE}
                    board[move_coords[0]][move_coords[1]] = current_player.letter
                    this_move.resultant_board_state = [row.copy() for row in board]
                    win_sets_with_this_square = filter(lambda win_set: move_coords in win_set, WIN_SETS)
                    for win_set in win_sets_with_this_square:
                        square_values = [board[x][y] for x, y in win_set]
                        is_winning_move = all(square == current_player.letter for square in square_values)
                        if is_winning_move:
                            this_move.outcomes |= {Outcome.WIN}
            except TimeoutError:
                this_move.outcomes |= {Outcome.TIMEOUT, Outcome.LOSE}

            self.moves.append(this_move)

        self.end_date = datetime.now()
