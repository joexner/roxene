import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional

from .constants import TF_PRECISION as PRECISION
from .persistence import EntityBase


class Cell(EntityBase):
    __tablename__ = "cell"
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    type: Mapped[str]

    __mapper_args__ = {
        "polymorphic_identity": "cell",
        "polymorphic_on": "type",
    }

    def get_output(self) -> PRECISION:
        pass


class InputCell(Cell):
    __tablename__ = "input_cell"
    __mapper_args__ = {"polymorphic_identity": "input"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"), primary_key=True)
    value: Mapped[Optional[float]] = mapped_column()


    def __init__(self, initial_value: PRECISION = None):
        self.id = uuid.uuid4()
        self.value = initial_value

    def set_output(self, value: PRECISION):
        self.value = value

    def get_output(self) -> PRECISION:
        return self.value
