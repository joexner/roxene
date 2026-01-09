from ..genes.create_input_cell import CreateInputCell
from ..mutagen import Mutagen
from ..util import wiggle, get_rng


class WiggleInputCellValue(Mutagen):
    """
    Mutagen that wiggles the initial_value of a CreateInputCell gene.
    
    This mutagen applies a small random perturbation to the initial value
    of input cells, allowing gradual evolution of input cell starting states.
    The severity parameter controls the magnitude of the wiggle.
    
    Attributes:
        severity: Controls the magnitude of the value change (default: 1.0)
                  Higher values cause larger changes to the initial_value.
    """
    __mapper_args__ = {"polymorphic_identity": "wiggle_input_cell_value"}

    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_CreateInputCell(self, gene: CreateInputCell) -> CreateInputCell:
        # Apply wiggle transformation to the initial value
        # Using severity to scale the wiggle parameters
        # Special handling for zero values since wiggle() uses log transform
        if gene.initial_value == 0.0:
            # For zero values, use only absolute wiggle to generate a new value
            new_value = get_rng().normal(0.0, self.severity * 0.1)
        else:
            new_value = wiggle(
                gene.initial_value,
                log_wiggle=self.severity * 0.5,
                absolute_wiggle=self.severity * 0.1
            )
        return CreateInputCell(new_value, parent_gene=gene)
