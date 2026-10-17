from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import torch

from sqlalchemy import ForeignKey, Integer, String, DateTime
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict

from .constants import NP_PRECISION
from .cell import Cell
from .cells.input_cell import InputCell
from .cells.neuron import Neuron
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
        self._compiled_execution = None
        for name in input_names:
            self.inputs[name] = InputCell()
        self.unused_output_names.extend(output_names)
        self.cells.extend(self.inputs.values())
        self.genotype = genotype
        if genotype:
            genotype.execute(self)

    def set_input(self, input_label, input_value):
        input_cell: InputCell = self.inputs[input_label]
        input_cell.set_output(input_value)
        compiled = getattr(self, "_compiled_execution", None)
        if compiled is not None and input_cell in compiled["input_cells_map"]:
            idx = compiled["input_cells_map"][input_cell]
            compiled["C"][0, idx] = input_value

    # TODO: Make output a @property, and maybe input and cells
    def get_output(self, output_label) -> NP_PRECISION:
        return self.outputs[output_label].get_output()

    def sync_neuron_inputs(self):
        compiled = getattr(self, "_compiled_execution", None)
        if compiled is not None:
            torch.matmul(compiled["C"], compiled["P_input"], out=compiled["all_inputs"])

    def update(self, sync_neuron_inputs: bool = True):
        compiled = getattr(self, "_compiled_execution", None)

        if compiled is None:
            neurons = [cell for cell in self.cells if isinstance(cell, Neuron)]
            if not neurons:
                return

            input_cells = [c for c in self.cells if isinstance(c, InputCell)]
            num_inputs = len(input_cells)
            rec_len = sum(n.feedback.shape[0] + 1 for n in neurons)

            unconnected_ports = []
            for n in neurons:
                for p in range(n.input.shape[0]):
                    if p not in n.bound_ports:
                        unconnected_ports.append((n, p))
            total_unconnected = len(unconnected_ports)
            dim_C = num_inputs + rec_len + total_unconnected

            cell_to_idx = {ic: i for i, ic in enumerate(input_cells)}
            rec_pos = num_inputs
            for n in neurons:
                rec_pos += n.feedback.shape[0]
                cell_to_idx[n] = rec_pos
                rec_pos += 1

            total_in_fb = sum(n.input.shape[0] + n.feedback.shape[0] for n in neurons)
            dtype = neurons[0].input.dtype
            device = neurons[0].input.device

            P_route = torch.zeros(dim_C, total_in_fb, dtype=dtype, device=device)
            in_fb_pos = 0
            rec_fb_pos = num_inputs
            unconn_pos = num_inputs + rec_len

            for n in neurons:
                for p in range(n.input.shape[0]):
                    if p in n.bound_ports:
                        P_route[cell_to_idx[n.bound_ports[p]], in_fb_pos + p] = 1.0
                    else:
                        P_route[unconn_pos, in_fb_pos + p] = 1.0
                        unconn_pos += 1
                in_fb_pos += n.input.shape[0]
                for fb_i in range(n.feedback.shape[0]):
                    P_route[rec_fb_pos + fb_i, in_fb_pos + fb_i] = 1.0
                rec_fb_pos += n.feedback.shape[0] + 1
                in_fb_pos += n.feedback.shape[0]

            w1_list = [torch.cat([n.input_hidden, n.feedback_hidden], dim=0) for n in neurons]
            w2_list = [torch.cat([n.hidden_feedback, n.hidden_output], dim=1) for n in neurons]
            w1 = torch.block_diag(*w1_list)
            w2 = torch.block_diag(*w2_list)
            W_eff1 = P_route @ w1

            total_ports = sum(n.input.shape[0] for n in neurons)
            P_input = torch.zeros(dim_C, total_ports, dtype=dtype, device=device)
            port_pos = 0
            unconn_pos = num_inputs + rec_len
            for n in neurons:
                for p in range(n.input.shape[0]):
                    if p in n.bound_ports:
                        P_input[cell_to_idx[n.bound_ports[p]], port_pos + p] = 1.0
                    else:
                        P_input[unconn_pos, port_pos + p] = 1.0
                        unconn_pos += 1
                port_pos += n.input.shape[0]

            C = torch.zeros(1, dim_C, dtype=dtype, device=device)
            for i, ic in enumerate(input_cells):
                val = ic.get_output()
                if val is not None:
                    C[0, i] = torch.tensor(val, dtype=dtype, device=device)
            rec_pos = num_inputs
            for n in neurons:
                fb_len = n.feedback.shape[0]
                C[0, rec_pos:rec_pos + fb_len] = n.feedback
                rec_pos += fb_len
                C[0, rec_pos:rec_pos + 1] = n.output
                rec_pos += 1

            unconn_idx = num_inputs + rec_len
            for n, p in unconnected_ports:
                C[0, unconn_idx] = n.input[p]
                unconn_idx += 1

            all_inputs = torch.zeros(1, total_ports, dtype=dtype, device=device)
            rec_pos = num_inputs
            port_pos = 0
            for n in neurons:
                fb_len = n.feedback.shape[0]
                n.feedback = C[0, rec_pos:rec_pos + fb_len]
                rec_pos += fb_len
                n.output = C[0, rec_pos:rec_pos + 1]
                rec_pos += 1

                in_len = n.input.shape[0]
                n.input = all_inputs[0, port_pos:port_pos + in_len]
                port_pos += in_len

            input_cells_map = {ic: i for i, ic in enumerate(input_cells)}

            compiled = {
                "id": self.id,
                "W_eff1": W_eff1,
                "w2": w2,
                "P_input": P_input,
                "all_inputs": all_inputs,
                "C": C,
                "num_inputs": num_inputs,
                "rec_len": rec_len,
                "input_cells_map": input_cells_map,
            }
            self._compiled_execution = compiled

        C = compiled["C"]
        W_eff1 = compiled["W_eff1"]
        w2 = compiled["w2"]
        num_inputs = compiled["num_inputs"]
        rec_len = compiled["rec_len"]

        hidden = torch.tanh(torch.matmul(C, W_eff1))
        if sync_neuron_inputs:
            torch.matmul(C, compiled["P_input"], out=compiled["all_inputs"])
        torch.tanh(torch.matmul(hidden, w2), out=C[:, num_inputs:num_inputs + rec_len])

    def addNeuron(self, neuron: Neuron):
        self._compiled_execution = None
        self.cells.insert(0, neuron)
        if self.unused_output_names:
            new_output_name = self.unused_output_names.pop()
            self.outputs[new_output_name] = neuron

    def __str__(self):
        return f"O-{str(self.id)[-7:]}"
