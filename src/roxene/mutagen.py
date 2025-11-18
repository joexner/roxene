import abc
import uuid
from typing import Optional
from numpy.random import Generator
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from .gene import Gene
from .genes.composite_gene import CompositeGene
from .genes.create_neuron import CreateNeuron
from .persistence import EntityBase
from .util import wiggle


class _Mutagen_Susceptibility(EntityBase):
    __tablename__ = "mutagen_susceptibility"

    mutagen_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("mutagen.id"), primary_key=True)
    gene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    susceptibility: Mapped[float]

    mutagen: Mapped["Mutagen"] = relationship(back_populates="_susceptibility_records")
    gene: Mapped[Optional[Gene]] = relationship()

    def __init__(self, gene: Optional[Gene], susceptibility: float):
        self.gene = gene
        self.susceptibility = susceptibility


class Mutagen(EntityBase):
    __tablename__ = "mutagen"
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    type: Mapped[str]
    susceptibility_log_wiggle: Mapped[float]

    __mapper_args__ = {
        "polymorphic_identity": "mutagen",
        "polymorphic_on": "type",
    }

    _susceptibility_records: Mapped[dict[Optional[Gene], _Mutagen_Susceptibility]] = relationship(
        collection_class=attribute_keyed_dict("gene"),
        cascade="all, delete-orphan",
        back_populates="mutagen"
    )

    susceptibilities: AssociationProxy[dict[Optional[Gene], float]] = association_proxy(
        target_collection="_susceptibility_records",
        attr="susceptibility",
        creator=lambda gene, susceptibility: _Mutagen_Susceptibility(gene, susceptibility)
    )

    def __init__(self, base_susceptibility: float, susceptibility_log_wiggle: float):
        self.id = uuid.uuid4()
        self.susceptibility_log_wiggle = susceptibility_log_wiggle
        # Initialize with base susceptibility for None gene
        self.susceptibilities[None] = base_susceptibility

    def get_mutation_susceptibility(self, gene: Gene, rng: Generator) -> float:
        if gene in self.susceptibilities:
            return self.susceptibilities[gene]
        
        parent_gene = getattr(gene, "parent_gene", None)
        parent_sus = self.get_mutation_susceptibility(parent_gene, rng)
        result = wiggle(parent_sus, rng, self.susceptibility_log_wiggle)
        
        # Store in database-backed dictionary
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
