from __future__ import annotations

import uuid
from sqlalchemy import CHAR, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, composite, relationship
from typing import List, Set, Optional

from .outcome import Outcome
from .persistence import Board, Point, OutcomeSet
from .players import Player
from ..persistence import EntityBase


class Move(EntityBase):
    __tablename__ = 'move'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)

    trial_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('trial.id'))
    trial: Mapped['Trial'] = relationship('Trial', back_populates='moves')

    letter: Mapped[str] = mapped_column(CHAR(1))
    initial_board_state: Mapped[List[List[str]]] = mapped_column(Board)
    position: Mapped[Point] = composite(mapped_column("row", nullable=True), mapped_column("column", nullable=True))
    resultant_board_state: Mapped[Optional[List[List[str]]]] = mapped_column(Board, nullable=True)

    outcomes: Mapped[Set[Outcome]] = mapped_column(OutcomeSet)

    # Human and fake players have no organism
    organism_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey('organism.id'))
    organism: Mapped[Optional['Organism']] = relationship('Organism')

    def __init__(self, player: Player, initial_board_state: List[List[str]]):
        self.id = uuid.uuid4()
        self.letter = player.letter
        self.initial_board_state = [row.copy() for row in initial_board_state]
        self.outcomes = set()
