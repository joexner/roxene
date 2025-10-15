from numpy import ndarray
import numpy as np
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from typing import Dict

import torch
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict

from ..cell import Cell
from ..constants import NP_PRECISION, TORCH_PRECISION
from ..persistence import TrackedTensor, WrappedTensor, EntityBase

activation_func = torch.tanh


class Neuron(Cell):
    __tablename__ = "neuron"
    __mapper_args__ = {"polymorphic_identity": "neuron"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"), primary_key=True)

    input: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    feedback: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    output: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    input_hidden: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    hidden_feedback: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    feedback_hidden: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))
    hidden_output: Mapped[torch.Tensor] = mapped_column(TrackedTensor.as_mutable(WrappedTensor))

    _ports_map: Mapped[Dict[int, "_Neuron_Input"]] = relationship(
        back_populates="listener_neuron",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("port"),
        lazy="joined")

    bound_ports: AssociationProxy[Dict[int, Cell]] = association_proxy(
        target_collection="_ports_map",
        attr="broadcaster_cell",
        creator=lambda port, inputcell: _Neuron_Input(port, inputcell))

    def __init__(self,
                 input: ndarray,
                 feedback: ndarray,
                 output: ndarray,
                 input_hidden: ndarray,
                 hidden_feedback: ndarray,
                 feedback_hidden: ndarray,
                 hidden_output: ndarray):
        '''
            I guess there's not much enforcement of this API anyway,
            but we're really trying to only take numpy ndarrays as inputs here.
            They should be the "right" sizes and types, but that depends on the Genes/tests
            that call this c'tor, or else you could build a Neuron that asplodes at runtime
        '''
        self.id = uuid.uuid4()
        self.input = torch.tensor(input, dtype=TORCH_PRECISION)
        self.feedback = torch.tensor(feedback, dtype=TORCH_PRECISION)
        self.output = torch.tensor(output, dtype=TORCH_PRECISION)

        self.input_hidden = torch.tensor(input_hidden, dtype=TORCH_PRECISION)
        self.hidden_feedback = torch.tensor(hidden_feedback, dtype=TORCH_PRECISION)
        self.feedback_hidden = torch.tensor(feedback_hidden, dtype=TORCH_PRECISION)
        self.hidden_output = torch.tensor(hidden_output, dtype=TORCH_PRECISION)
        self.bound_ports = {}

    def update(self) -> None:
        # TODO: Optimize / make less wack
        for port_num, cell in self.bound_ports.items():
            self.input[port_num] = cell.get_output()
        
        hidden_in = torch.cat([self.input, self.feedback], dim=0).unsqueeze(0)
        hidden_wts = torch.cat([self.input_hidden, self.feedback_hidden], dim=0)
        hidden = activation_func(torch.matmul(hidden_in, hidden_wts))
        
        self.feedback.copy_(activation_func(torch.matmul(hidden, self.hidden_feedback)).squeeze(0))
        self.output.copy_(activation_func(torch.matmul(hidden, self.hidden_output)).squeeze(0))

    def get_output(self) -> NP_PRECISION:
        return NP_PRECISION(self.output.item())

    def add_input_connection(self, tx_cell: Cell, req_port: int):
        num_ports = self.input.shape[0]
        for offset in range(num_ports):
            rx_port = (req_port + offset) % num_ports
            if rx_port not in self.bound_ports:
                self.bound_ports[rx_port] = tx_cell
                return

    def __str__(self):
        return f"N-{str(self.id)[-7:]}"


class _Neuron_Input(EntityBase):
    __tablename__ = "neuron_input"

    listener_neuron_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("neuron.id"), primary_key=True)
    port: Mapped[int] = mapped_column(primary_key=True)

    broadcaster_cell_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"))

    listener_neuron: Mapped[Neuron] = relationship(foreign_keys=[listener_neuron_id])
    broadcaster_cell: Mapped[Cell] = relationship(foreign_keys=[broadcaster_cell_id], lazy="immediate")

    def __init__(self, port: int, inputcell: Cell):
        self.port = port
        self.broadcaster_cell = inputcell
