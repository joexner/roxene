import uuid
from enum import IntEnum

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from roxene import Gene, Organism


class RotateCells(Gene):
    __tablename__ = "rotate_cells"
    __mapper_args__ = {"polymorphic_identity": "rotate_cells"}

    class Direction(IntEnum):
        FORWARD = 1
        BACKWARD = -1

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    direction: Mapped[Direction] = mapped_column(Integer)

    def __init__(self, direction: Direction = Direction.BACKWARD, parent_gene=None):
        super().__init__(parent_gene)
        self.direction = direction

    def execute(self, organism: Organism):
        if self.direction is RotateCells.Direction.FORWARD:
            popped = organism.cells.pop()
            organism.cells.insert(0, popped)
        elif self.direction is RotateCells.Direction.BACKWARD:
            popped = organism.cells.pop(0)
            organism.cells.append(popped)
