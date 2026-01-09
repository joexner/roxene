from sqlalchemy.orm import Mapped, synonym

from ..gene import Gene
from ..genes.rotate_cells import RotateCells
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene


class AddRotateCells(AddGene):
    __mapper_args__ = {"polymorphic_identity": "add_rotate_cells"}

    direction: Mapped[int] = synonym("_i1")

    def __init__(self, base_susceptibility: float = 0.01, direction: RotateCells.Direction = RotateCells.Direction.BACKWARD):
        super().__init__(base_susceptibility)
        self.direction = direction

    def get_new_gene(self, parent_gene: CompositeGene) -> Gene | None:
        return RotateCells(RotateCells.Direction(self.direction))
