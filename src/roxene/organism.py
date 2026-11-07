from __future__ import annotations

import itertools
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import ForeignKey, Integer, String, DateTime, event
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, attribute_keyed_dict

from .constants import NUMPY_BACKEND, NUMPY_BACKEND_MAX_FLOPS, NP_PRECISION
from .cell import Cell
from .cells.input_cell import InputCell
from .cells.neuron import Neuron
from .compiled_network import CompiledNetwork, NumpyCompiledNetwork
from .gene import Gene
from .persistence import EntityBase


class _Organism_Input(EntityBase):
    __tablename__ = "organism_input"

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id", ondelete="CASCADE"), primary_key=True)
    name: Mapped[str] = mapped_column(primary_key=True)
    inputcell_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("input_cell.id"))

    organism: Mapped["Organism"] = relationship()
    inputcell: Mapped[InputCell] = relationship(lazy="joined")

    def __init__(self, name, inputcell):
        self.name = name
        self.inputcell = inputcell


class _Organism_Output(EntityBase):
    __tablename__ = "organism_output"

    organism_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organism.id", ondelete="CASCADE"), primary_key=True)
    name: Mapped[str] = mapped_column(primary_key=True)
    neuron_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("neuron.id"))

    organism: Mapped["Organism"] = relationship()
    neuron: Mapped[Neuron] = relationship(lazy="joined")

    def __init__(self, name, neuron):
        self.name = name
        self.neuron = neuron


class _Organism_Cell(EntityBase):
    __tablename__ = "organism_cell"

    organism_id = mapped_column(ForeignKey("organism.id", ondelete="CASCADE"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)
    cell_id = mapped_column(ForeignKey("cell.id"), primary_key=True)

    organism: Mapped["Organism"] = relationship()
    cell: Mapped[Cell] = relationship(cascade="all", lazy="joined")

    def __init__(self, cell):
        self.cell = cell


class _Organism_Unused_Output_Name(EntityBase):
    __tablename__ = "organism_unused_output_name"

    organism_id = mapped_column(ForeignKey("organism.id", ondelete="CASCADE"), primary_key=True)
    ordinal = mapped_column("ordinal", Integer, primary_key=True)

    name = mapped_column(String())
    organism: Mapped["Organism"] = relationship()

    def __init__(self, name):
        self.name = name


def _estimate_flops(cells) -> int:
    """dim_C * total_in_fb for these cells, the size measure the numpy threshold uses."""
    neurons = [c for c in cells if isinstance(c, Neuron)]
    if not neurons:
        return 0
    dim_C = (sum(1 for c in cells if isinstance(c, InputCell))
             + sum(n.feedback.shape[0] + 1 for n in neurons)
             + sum(1 for n in neurons for p in range(n.input.shape[0]) if p not in n.bound_ports))
    total_in_fb = sum(n.input.shape[0] + n.feedback.shape[0] for n in neurons)
    return dim_C * total_in_fb


class Organism(EntityBase):
    __tablename__ = "organism"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    created_date: Mapped[datetime] = mapped_column(DateTime)
    deleted_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    cells: Mapped[List[Cell]] = association_proxy(target_collection="_cells_list", attr="cell", creator=_Organism_Cell)
    inputs: Mapped[Dict[str, InputCell]] = association_proxy(target_collection="_inputs_map", attr="inputcell", creator=_Organism_Input)
    outputs: Mapped[Dict[str, Neuron]] = association_proxy(target_collection="_outputs_map", attr="neuron", creator=_Organism_Output)

    genotype: Mapped[Optional[Gene]] = relationship()
    unused_output_names: Mapped[List[str]] = association_proxy(target_collection="_unused_output_names_list", attr="name", creator=_Organism_Unused_Output_Name)

    genotype_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("gene.id"))

    _inputs_map: Mapped[Dict[str, _Organism_Input]] = relationship(
        back_populates="organism",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("name"),
        lazy="joined",
    )

    _outputs_map: Mapped[Dict[str, _Organism_Output]] = relationship(
        back_populates="organism",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("name"),
        lazy="joined",
    )

    _cells_list: Mapped[List[_Organism_Cell]] = relationship(
        back_populates="organism",
        order_by="_Organism_Cell.ordinal",
        collection_class=ordering_list('ordinal'),
        cascade="all, delete-orphan",
        lazy="joined",
    )

    _unused_output_names_list: Mapped[List[_Organism_Unused_Output_Name]] = relationship(
        back_populates="organism",
        order_by="_Organism_Unused_Output_Name.ordinal",
        collection_class=ordering_list('ordinal'),
        cascade="all, delete-orphan",
        lazy="joined",
    )

    def __init__(self, input_names={}, output_names={}, genotype: Gene = None):
        self.id = uuid.uuid4()
        self.created_date = datetime.now()
        self._compiled_network = None
        for name in input_names:
            self.inputs[name] = InputCell()
        self.unused_output_names.extend(output_names)
        self.cells.extend(self.inputs.values())
        self.genotype = genotype
        if genotype:
            genotype.execute(self)

    def network(self) -> Optional[CompiledNetwork]:
        """The live CompiledNetwork, compiled on first use. None if there are no neurons."""
        if getattr(self, "_compiled_network", None) is None:
            self._compiled_network = self.compile()
        return self._compiled_network

    def compile(self) -> Optional[CompiledNetwork]:
        """Build the network: torch by default, the numpy fast path for small organisms."""
        if not any(isinstance(c, Neuron) for c in self.cells):
            return None
        if (NUMPY_BACKEND in ("on", "auto")
                and NUMPY_BACKEND_MAX_FLOPS > 0
                and _estimate_flops(self.cells) <= NUMPY_BACKEND_MAX_FLOPS):
            return NumpyCompiledNetwork(self.cells)
        return CompiledNetwork(self.cells)

    def set_input(self, input_label, input_value):
        input_cell: InputCell = self.inputs[input_label]
        input_cell.set_output(input_value)
        network = self.network()
        if network is not None:
            network.set_input(input_cell, input_value)

    # TODO: Make output a @property, and maybe input and cells
    def get_output(self, output_label) -> NP_PRECISION:
        neuron = self.outputs[output_label]
        network = self.network()
        if network is not None:
            value = network.get_output(neuron)
            if value is not None:
                return value
        return neuron.get_output()

    def sync_neuron_inputs(self):
        network = self.network()
        if network is not None:
            network.sync_neuron_inputs()

    def commit_state(self):
        """Sync the live compiled state (if any) back into the plain neuron arrays."""
        network = getattr(self, "_compiled_network", None)
        if network is not None:
            network.commit_state()

    def update(self, sync_neuron_inputs: bool = True):
        network = self.network()
        if network is not None:
            network.update(sync_neuron_inputs)

    def addNeuron(self, neuron: Neuron):
        self._compiled_network = None
        self.cells.insert(0, neuron)
        if self.unused_output_names:
            new_output_name = self.unused_output_names.pop()
            self.outputs[new_output_name] = neuron

    def __str__(self):
        return f"O-{str(self.id)[-7:]}"


@event.listens_for(Session, "before_flush")
def _commit_compiled_state(session, flush_context, instances):
    """Never let the flush serialize stale plain arrays while a CompiledNetwork is live."""
    for obj in itertools.chain(session.new, session.dirty):
        if isinstance(obj, Organism):
            obj.commit_state()
