from sqlalchemy.orm import Mapped, synonym

from ..gene import Gene
from ..genes.create_input_cell import CreateInputCell
from ..genes.composite_gene import CompositeGene
from .add_gene import AddGene


class AddCreateInputCell(AddGene):
    """
    Mutagen that adds a CreateInputCell gene to a CompositeGene.
    
    CreateInputCell genes create input cells that allow external values to be
    fed into the organism. This mutagen allows evolution to add new input
    pathways to an organism's genotype.
    
    Attributes:
        initial_value: The initial value for the new input cell (default: 0.0)
    """
    __mapper_args__ = {"polymorphic_identity": "add_create_input_cell"}

    # Store initial_value as an integer (scaled by 1000) in _i1 for persistence
    # We use _i1 for storage since it's an integer column
    _stored_initial_value: Mapped[int] = synonym("_i1")

    def __init__(self, base_susceptibility: float = 0.01, initial_value: float = 0.0):
        super().__init__(base_susceptibility)
        # Store as scaled integer for database persistence
        self._stored_initial_value = int(initial_value * 1000)

    @property
    def initial_value(self) -> float:
        """Get the initial value for new input cells."""
        if self._stored_initial_value is not None:
            return self._stored_initial_value / 1000.0
        return 0.0

    def get_new_gene(self, parent_gene: CompositeGene) -> Gene | None:
        return CreateInputCell(self.initial_value)
