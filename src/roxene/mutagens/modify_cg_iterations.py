from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyCGIterations(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_iterations"}

    def __init__(self, severity: float = 1.0, base_susceptibility: float = 0.01):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        max_delta = max(1, int(self.severity * parent_gene.iterations))
        # Always increase if at zero, otherwise random direction
        sign = 1 if parent_gene.iterations == 0 else get_rng().choice([-1, 1])
        delta = get_rng().integers(1, max_delta + 1) * sign
        return CompositeGene(parent_gene.child_genes, parent_gene.iterations + delta, parent_gene)
