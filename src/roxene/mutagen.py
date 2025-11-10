from abc import ABC
import uuid
from numpy.random import Generator
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .gene import Gene
from .genes.composite_gene import CompositeGene
from .genes.create_neuron import CreateNeuron
from .persistence import EntityBase
from .util import wiggle


class Mutagen(EntityBase, ABC):
    __tablename__ = "mutagen"
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    type: Mapped[str]
    base_susceptibility: Mapped[float]
    susceptibility_log_wiggle: Mapped[float]

    __mapper_args__ = {
        "polymorphic_identity": "mutagen",
        "polymorphic_on": "type",
    }

    def __init__(self, base_susceptibility: float, susceptibility_log_wiggle: float):
        self.id = uuid.uuid4()
        self.base_susceptibility = base_susceptibility
        self.susceptibility_log_wiggle = susceptibility_log_wiggle
        # susceptibilities is runtime state - not persisted
        self.susceptibilities = {None: base_susceptibility}

    def get_mutation_susceptibility(self, gene: Gene, rng: Generator) -> float:
        result = self.susceptibilities.get(gene)
        if result is None:
            parent_gene = getattr(gene, "parent_gene", None)
            parent_sus = self.get_mutation_susceptibility(parent_gene, rng)
            result = wiggle(parent_sus, rng, self.susceptibility_log_wiggle)
            self.susceptibilities[gene] = result
        return result

    def mutate(self, gene: Gene, rng: Generator) -> Gene:
        if isinstance(gene, CompositeGene):
            return self.mutate_CompositeGene(gene, rng)
        elif isinstance(gene, CreateNeuron):
            return self.mutate_CreateNeuron(gene, rng)
        else:
            return gene

    def mutate_CompositeGene(self, parent_gene: CompositeGene, rng):
        any_changed = False
        new_genes = []
        for orig in parent_gene.child_genes:
            mutant = self.mutate(orig, rng)
            new_genes.append(mutant)
            any_changed |= (mutant is not orig)
        if any_changed:
            return CompositeGene(new_genes, parent_gene.iterations, parent_gene)
        else:
            return parent_gene

    def mutate_CreateNeuron(self, gene: CreateNeuron, rng: Generator):
        return gene

    def __str__(self):
        return f"M-{str(self.id)[-7:]}"
