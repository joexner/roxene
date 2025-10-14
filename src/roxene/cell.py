import uuid

from sqlalchemy.orm import Mapped, mapped_column

from .constants import NP_PRECISION
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

    def get_output(self) -> NP_PRECISION:
        pass
