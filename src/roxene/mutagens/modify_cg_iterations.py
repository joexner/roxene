from ..genes.composite_gene import CompositeGene
from ..mutagen import Mutagen
from ..util import get_rng


class ModifyCGIterations(Mutagen):
    __mapper_args__ = {"polymorphic_identity": "modify_iterations"}

    def __init__(self, base_susceptibility: float = 0.01, severity: float = 1.0):
        super().__init__(base_susceptibility)
        self.severity = severity

    def mutate_CompositeGene(self, parent_gene: CompositeGene) -> CompositeGene:
        max_delta = max(1, int(self.severity * parent_gene.iterations))
        delta = int(get_rng().integers(-max_delta, max_delta + 1))
        new_iterations = max(0, parent_gene.iterations + delta)
        return CompositeGene(parent_gene.child_genes, new_iterations, parent_gene)
