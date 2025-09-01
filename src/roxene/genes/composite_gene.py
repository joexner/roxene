import uuid
from typing import List

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Mapped, mapped_column, relationship

from roxene import Gene, Organism, EntityBase


class CompositeGene(Gene):
    __tablename__ = "composite_gene"
    __mapper_args__ = {"polymorphic_identity": "composite_gene"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    child_genes: Mapped[List[Gene]] = association_proxy(target_collection="_genes_list", attr="child")
    iterations: Mapped[int] = mapped_column()

    _genes_list: Mapped[List["_CompositeGene_Child"]] = relationship(
        back_populates="gene",
        cascade="all, delete-orphan",
        collection_class=ordering_list('ordinal'),
        lazy="select",
    )

    def __init__(self, child_genes: List[Gene], iterations: int = 1, parent_gene=None):
        super().__init__(parent_gene)
        self.child_genes = child_genes
        self.iterations = iterations

    def execute(self, organism: Organism):
        for n in range(self.iterations):
            for gene in self.child_genes:
                gene.execute(organism)


class _CompositeGene_Child(EntityBase):
    __tablename__ = "composite_gene_child"

    gene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("composite_gene.id"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)
    child_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"))

    gene: Mapped[CompositeGene] = relationship(foreign_keys=[gene_id])
    child: Mapped[Gene] = relationship(foreign_keys=[child_id])

    def __init__(self, child: Gene):
        self.child = child
