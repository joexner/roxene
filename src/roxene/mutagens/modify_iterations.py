import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyIterations(Mutagen):
    """
    Modifies the iteration count of a CompositeGene.
    Can increase or decrease the number of times child genes are executed.
    """
    __tablename__ = "modify_iterations_mutagen"
    __mapper_args__ = {"polymorphic_identity": "modify_iterations_mutagen"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)

    def __init__(self, base_susceptibility: float = 0.01, susceptibility_log_wiggle: float = 0.01):
        super().__init__(base_susceptibility, susceptibility_log_wiggle)

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        # Check if this gene should be mutated based on susceptibility
        susceptibility = self.get_mutation_susceptibility(parent_gene)
        if get_rng().random() >= susceptibility:
            # No mutation, just recursively mutate child genes
            return super().mutate_CompositeGene(parent_gene)

        # Recursively mutate child genes
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)

        # Modify the iteration count by incrementing or decrementing by 1
        if get_rng().random() < 0.5:
            # Increment by 1
            new_iterations = parent_gene.iterations + 1
        else:
            # Decrement by 1, but never go below 0
            new_iterations = max(0, parent_gene.iterations - 1)

        if new_iterations != parent_gene.iterations:
            any_changed = True

        if any_changed:
            return CompositeGene(new_genes, new_iterations, parent_gene)
        else:
            return parent_gene
