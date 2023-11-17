from numpy import ndarray
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy
from typing import Dict

import tensorflow as tf
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, attribute_keyed_dict

from .cells import Cell
from .constants import TF_PRECISION as PRECISION
from .persistence import TrackedVariable, WrappedVariable, WrappedTensor, EntityBase

activation_func = tf.nn.tanh


class Neuron(Cell):
    __tablename__ = "neuron"
    __mapper_args__ = {"polymorphic_identity": "neuron"}

    id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"), primary_key=True)

    input: Mapped[tf.Variable] = mapped_column(TrackedVariable.as_mutable(WrappedVariable))
    feedback: Mapped[tf.Variable] = mapped_column(TrackedVariable.as_mutable(WrappedVariable))
    output: Mapped[tf.Variable] = mapped_column(TrackedVariable.as_mutable(WrappedVariable))
    input_hidden: Mapped[tf.Tensor] = mapped_column(WrappedTensor)
    hidden_feedback: Mapped[tf.Tensor] = mapped_column(WrappedTensor)
    feedback_hidden: Mapped[tf.Tensor] = mapped_column(WrappedTensor)
    hidden_output: Mapped[tf.Tensor] = mapped_column(WrappedTensor)

    _ports_map: Mapped[Dict[int, "_Neuron_Input"]] = relationship(
        back_populates="listener_neuron",
        cascade="all, delete-orphan",
        collection_class=attribute_keyed_dict("port"))

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
        self.input = tf.Variable(initial_value=input, dtype=PRECISION)
        self.feedback = tf.Variable(initial_value=feedback, dtype=PRECISION)
        self.output = tf.Variable(initial_value=output, dtype=PRECISION)

        self.input_hidden = tf.convert_to_tensor(input_hidden, dtype=PRECISION)
        self.hidden_feedback = tf.convert_to_tensor(hidden_feedback, dtype=PRECISION)
        self.feedback_hidden = tf.convert_to_tensor(feedback_hidden, dtype=PRECISION)
        self.hidden_output = tf.convert_to_tensor(hidden_output, dtype=PRECISION)
        self.bound_ports = {}

    def update(self) -> None:
        # TODO: Optimize / make less wack
        if len(self.bound_ports) > 0:
            ports = [[n] for n in self.bound_ports.keys()]
            values = [self.bound_ports[port[0]].get_output() for port in ports]
            self.input.assign(tf.tensor_scatter_nd_update(self.input, ports, values))
        hidden_in = tf.expand_dims(tf.concat([self.input, self.feedback], 0), 0)
        hidden_wts = tf.concat([self.input_hidden, self.feedback_hidden], 0)
        hidden = activation_func(tf.matmul(hidden_in, hidden_wts))
        self.feedback.assign(tf.squeeze(activation_func(tf.matmul(hidden, self.hidden_feedback)), 0))
        self.output.assign(tf.squeeze(activation_func(tf.matmul(hidden, self.hidden_output)), [0]))

    def get_output(self) -> PRECISION:
        return self.output.numpy()

    def add_input_connection(self, tx_cell: Cell, req_port: int):
        num_ports = self.input.shape[0]
        for offset in range(num_ports):
            rx_port = (req_port + offset) % num_ports
            if rx_port not in self.bound_ports:
                self.bound_ports[rx_port] = tx_cell
                return


class _Neuron_Input(EntityBase):
    __tablename__ = "neuron_input"

    listener_neuron_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("neuron.id"), primary_key=True)
    port: Mapped[int] = mapped_column(primary_key=True)

    broadcaster_cell_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cell.id"))

    listener_neuron: Mapped[Neuron] = relationship(foreign_keys=[listener_neuron_id])
    broadcaster_cell: Mapped[Cell] = relationship(foreign_keys=[broadcaster_cell_id])

    def __init__(self, port: int, inputcell: Cell):
        self.port = port
        self.broadcaster_cell = inputcell
