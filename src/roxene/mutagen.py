import uuid
from typing import Optional

from sqlalchemy import ForeignKey, Integer, Float
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict, validates

from .gene import Gene
from .genes.composite_gene import CompositeGene
from .genes.connect_neurons import ConnectNeurons
from .genes.create_neuron import CreateNeuron
from .genes.rotate_cells import RotateCells
from .persistence import EntityBase
from .util import wiggle, get_rng

# Constant for susceptibility log wiggle used across all mutagens
SUSCEPTIBILITY_LOG_WIGGLE = 0.01


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

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    type: Mapped[str]

    severity: Mapped[Optional[float]] = mapped_column("f1", Float, nullable=True)

    base_susceptibility: Mapped[float]

    # Columns for subclass use - not used in base Mutagen
    _i1: Mapped[Optional[int]] = mapped_column("i1", Integer, nullable=True)
    _i2: Mapped[Optional[int]] = mapped_column("i2", Integer, nullable=True)
    _i3: Mapped[Optional[int]] = mapped_column("i3", Integer, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "mutagen",
        "polymorphic_on": "type",
    }

    _susceptibility_records: Mapped[dict[Gene, _Mutagen_Susceptibility]] = relationship(
        collection_class=attribute_keyed_dict("gene"),
        cascade="all, delete-orphan",
        back_populates="mutagen"
    )

    susceptibilities: AssociationProxy[dict[Gene, float]] = association_proxy(
        target_collection="_susceptibility_records",
        attr="susceptibility",
        creator=_Mutagen_Susceptibility
    )

    def __init__(self, base_susceptibility: float):
        self.id = uuid.uuid4()
        self.base_susceptibility = base_susceptibility

    @validates("severity", "base_susceptibility")
    def validate_range(self, key, value):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError(f"{key} must be between 0.0 and 1.0, got {value}")
        return value

    def get_mutation_susceptibility(self, gene: Gene) -> float:
        result = self.susceptibilities.get(gene)
        if result is None:
            parent_gene = getattr(gene, "parent_gene", None)
            if parent_gene is None:
                result = self.base_susceptibility
            else:
                parent_sus = self.get_mutation_susceptibility(parent_gene)
                result = wiggle(parent_sus, SUSCEPTIBILITY_LOG_WIGGLE)
            self.susceptibilities[gene] = result
        return result
    
    def should_mutate(self, gene: Gene) -> bool:
        susceptibility = self.get_mutation_susceptibility(gene)
        return get_rng().random() < susceptibility

    def mutate(self, gene: Gene) -> Gene:
        if isinstance(gene, CompositeGene):
            any_changed = False
            new_genes = []
            for child in gene.child_genes:
                mutant = self.mutate(child)
                new_genes.append(mutant)
                any_changed |= (mutant is not child)
            if any_changed:
                gene = CompositeGene(new_genes, gene.iterations, gene)
            return self.mutate_CompositeGene(gene) if self.should_mutate(gene) else gene
        elif isinstance(gene, CreateNeuron):
            return self.mutate_CreateNeuron(gene) if self.should_mutate(gene) else gene
        elif isinstance(gene, ConnectNeurons):
            return self.mutate_ConnectNeurons(gene) if self.should_mutate(gene) else gene
        elif isinstance(gene, RotateCells):
            return self.mutate_RotateCells(gene) if self.should_mutate(gene) else gene
        else:
            return gene

    def mutate_CompositeGene(self, gene: CompositeGene):
        return gene

    def mutate_CreateNeuron(self, gene: CreateNeuron):
        return gene

    def mutate_ConnectNeurons(self, gene: ConnectNeurons):
        return gene

    def mutate_RotateCells(self, gene: RotateCells):
        return gene
