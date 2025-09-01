import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from roxene import Gene, Organism, InputCell


class CreateInputCell(Gene):
    __tablename__ = "create_input_cell"
    __mapper_args__ = {"polymorphic_identity": "create_input_cell"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("gene.id"), primary_key=True)
    initial_value: Mapped[float] = mapped_column()

    def __init__(self, initial_value, parent_gene=None):
        super().__init__(parent_gene)
        self.initial_value = initial_value

    def execute(self, organism: Organism):
        input_cell = InputCell(self.initial_value)
        organism.cells.insert(0, input_cell)
