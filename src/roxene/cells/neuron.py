import numpy as np

from numpy import ndarray
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from typing import Dict

import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict

from ..cell import Cell
from ..constants import NP_PRECISION
from ..persistence import FloatArrayType, EntityBase


class Neuron(Cell):
    __tablename__ = "neuron"
    __mapper_args__ = {"polymorphic_identity": "neuron"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"), primary_key=True)

    input: Mapped[ndarray] = mapped_column(FloatArrayType)
    feedback: Mapped[ndarray] = mapped_column(FloatArrayType)
    output: Mapped[ndarray] = mapped_column(FloatArrayType)
    input_hidden: Mapped[ndarray] = mapped_column(FloatArrayType)
    hidden_feedback: Mapped[ndarray] = mapped_column(FloatArrayType)
    feedback_hidden: Mapped[ndarray] = mapped_column(FloatArrayType)
    hidden_output: Mapped[ndarray] = mapped_column(FloatArrayType)

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
        self.input = np.array(input, dtype=NP_PRECISION, copy=True)
        self.feedback = np.array(feedback, dtype=NP_PRECISION, copy=True)
        self.output = np.array(output, dtype=NP_PRECISION, copy=True)

        self.input_hidden = np.array(input_hidden, dtype=NP_PRECISION, copy=True)
        self.hidden_feedback = np.array(hidden_feedback, dtype=NP_PRECISION, copy=True)
        self.feedback_hidden = np.array(feedback_hidden, dtype=NP_PRECISION, copy=True)
        self.hidden_output = np.array(hidden_output, dtype=NP_PRECISION, copy=True)
        self.bound_ports = {}

    def get_output(self) -> NP_PRECISION:
        return NP_PRECISION(float(self.output[0]))

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
