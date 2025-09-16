from __future__ import annotations

import abc
import uuid
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .persistence import EntityBase


class Gene(EntityBase):
    __tablename__ = "gene"
    __allow_unmapped__ = True

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    parent_gene_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("gene.id"))
    parent_gene: Mapped[Gene] = relationship("Gene", remote_side=[id])
    type: Mapped[str]

    __mapper_args__ = {
        "polymorphic_identity": "gene",
        "polymorphic_on": "type",
    }

    def __init__(self, parent_gene=None):
        self.parent_gene = parent_gene
        self.id = uuid.uuid4()

    @abc.abstractmethod
    def execute(self, organism: 'Organism'):
        pass

    def __str__(self):
        return f"G-{str(self.id)[-7:]}"