import uuid
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column, Mapped

from ..constants import NP_PRECISION
from ..cell import Cell


class InputCell(Cell):
    __tablename__ = "input_cell"
    __mapper_args__ = {"polymorphic_identity": "input"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"), primary_key=True)
    value: Mapped[Optional[float]] = mapped_column()


    def __init__(self, initial_value: NP_PRECISION = None):
        self.id = uuid.uuid4()
        self.value = initial_value

    def set_output(self, value: NP_PRECISION):
        self.value = value

    def get_output(self) -> NP_PRECISION:
        return self.value
