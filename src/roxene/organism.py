from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
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

    # TODO: Make output a @property, and maybe input and cells
    def get_output(self, output_label) -> NP_PRECISION:
        return self.outputs[output_label].get_output()

    def update(self):
        compiled = getattr(self, "_compiled_execution", None)

        if compiled is None:
            neurons = [cell for cell in self.cells if isinstance(cell, Neuron)]
            if not neurons:
                return

            w1_list = [torch.cat([n.input_hidden, n.feedback_hidden], dim=0) for n in neurons]
            w2_list = [torch.cat([n.hidden_feedback, n.hidden_output], dim=1) for n in neurons]
            w1 = torch.block_diag(*w1_list)
            w2 = torch.block_diag(*w2_list)

            total_in_fb = sum(n.input.shape[0] + n.feedback.shape[0] for n in neurons)
            in_fb = torch.zeros(1, total_in_fb, dtype=w1.dtype, device=w1.device)

            in_slices = []
            fb_slices = []
            pos = 0
            for n in neurons:
                in_len = n.input.shape[0]
                fb_len = n.feedback.shape[0]
                in_slices.append((pos, pos + in_len))
                in_fb[0, pos:pos + in_len] = n.input
                pos += in_len
                fb_slices.append((pos, pos + fb_len))
                in_fb[0, pos:pos + fb_len] = n.feedback
                pos += fb_len

            res_fb_slices = []
            res_out_indices = []
            res_pos = 0
            for n in neurons:
                fb_len = n.feedback.shape[0]
                res_fb_slices.append((res_pos, res_pos + fb_len))
                res_pos += fb_len
                res_out_indices.append(res_pos)
                res_pos += 1

            compiled = {
                "id": self.id,
                "neurons": neurons,
                "w1": w1,
                "w2": w2,
                "in_fb": in_fb,
                "in_slices": in_slices,
                "fb_slices": fb_slices,
                "res_fb_slices": res_fb_slices,
                "res_out_indices": res_out_indices,
            }
            self._compiled_execution = compiled

        neurons = compiled["neurons"]
        in_fb = compiled["in_fb"]
        w1 = compiled["w1"]
        w2 = compiled["w2"]
        in_slices = compiled["in_slices"]
        fb_slices = compiled["fb_slices"]
        res_fb_slices = compiled["res_fb_slices"]
        res_out_indices = compiled["res_out_indices"]

        tx_outputs = {cell: cell.get_output() for cell in self.cells}
        for i, n in enumerate(neurons):
            in_s, _ = in_slices[i]
            for port_num, src_cell in n.bound_ports.items():
                val = tx_outputs[src_cell]
                if isinstance(val, (np.ndarray, np.generic)):
                    val = torch.tensor(val, dtype=n.input.dtype, device=n.input.device)
                in_fb[0, in_s + port_num] = val
                n.input[port_num] = val

        hidden = torch.tanh(torch.matmul(in_fb, w1))
        res = torch.tanh(torch.matmul(hidden, w2)).squeeze(0)

        for i, n in enumerate(neurons):
            rf_s, rf_e = res_fb_slices[i]
            fb_s, fb_e = fb_slices[i]
            fb_val = res[rf_s:rf_e]
            in_fb[0, fb_s:fb_e] = fb_val
            n.feedback.copy_(fb_val)

            out_idx = res_out_indices[i]
            out_val = res[out_idx:out_idx + 1]
            n.output.copy_(out_val)

    def addNeuron(self, neuron: Neuron):
        self._compiled_execution = None
        self.cells.insert(0, neuron)
        if self.unused_output_names:
            new_output_name = self.unused_output_names.pop()
            self.outputs[new_output_name] = neuron

    def __str__(self):
        return f"O-{str(self.id)[-7:]}"
