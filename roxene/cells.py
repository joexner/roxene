import abc
import tensorflow as tf
import uuid
from numpy import ndarray
from sqlalchemy.orm import Mapped, mapped_column
from typing import Dict, Optional

from constants import TF_PRECISION as PRECISION
from persistence import EntityBase


class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self) -> PRECISION:
        pass


activation_func = tf.nn.tanh

class Neuron(Cell):

    input: tf.Variable
    feedback: tf.Variable
    output: tf.Variable
    input_hidden: tf.Tensor
    hidden_feedback: tf.Tensor
    feedback_hidden: tf.Tensor
    hidden_output: tf.Tensor

    input_ports: Dict[int, Cell]

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
        self.input = tf.Variable(initial_value=input, dtype=PRECISION)
        self.feedback = tf.Variable(initial_value=feedback, dtype=PRECISION)
        self.output = tf.Variable(initial_value=output, dtype=PRECISION)

        self.input_hidden = tf.convert_to_tensor(input_hidden, dtype=PRECISION)
        self.hidden_feedback = tf.convert_to_tensor(hidden_feedback, dtype=PRECISION)
        self.feedback_hidden = tf.convert_to_tensor(feedback_hidden, dtype=PRECISION)
        self.hidden_output = tf.convert_to_tensor(hidden_output, dtype=PRECISION)
        self.input_ports = {}

    def update(self) -> None:
        # TODO: Optimize / make less wack
        if len(self.input_ports) > 0:
            ports = [[n] for n in self.input_ports.keys()]
            values = [self.input_ports[port[0]].get_output() for port in ports]
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
            if rx_port not in self.input_ports:
                self.input_ports[rx_port] = tx_cell
                return


class InputCell(EntityBase):
    __tablename__ = "input_cell"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    value: Mapped[Optional[float]] = mapped_column()


    def __init__(self, initial_value: PRECISION = None):
        self.id = uuid.uuid4()
        self.value = initial_value

    def set_output(self, value: PRECISION):
        self.value = value

    def get_output(self) -> PRECISION:
        return self.value
