from ..genes.rotate_cells import RotateCells
from ..mutagen import Mutagen


class ChangeRotateDirection(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "change_rotate_direction"}

    def __init__(self, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)

    def mutate_RotateCells(self, gene: RotateCells) -> RotateCells:
        # Flip the direction: FORWARD <-> BACKWARD
        new_direction = (
            RotateCells.Direction.BACKWARD
            if gene.direction == RotateCells.Direction.FORWARD
            else RotateCells.Direction.FORWARD
        )
        return RotateCells(new_direction, parent_gene=gene)
