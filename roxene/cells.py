import abc

import tensorflow as tf
from numpy import ndarray

from .constants import TF_PRECISION as PRECISION


class Cell(abc.ABC):

    @abc.abstractmethod
    def get_output(self) -> PRECISION:
        pass


class Neuron(Cell):
    ACTIVATION = tf.nn.tanh

    def __init__(self,
                 input_initial_value: ndarray,
                 feedback_initial_value: ndarray,
                 output_initial_value: ndarray,
                 input_hidden: ndarray,
                 hidden_feedback: ndarray,
                 feedback_hidden: ndarray,
                 hidden_output: ndarray,
                 activation=ACTIVATION,
                 precision=PRECISION):
        '''
            I guess there's not much enforcement of this API anyway,
            but we're really trying to only take numpy ndarrays as inputs here.
            They should be the "right" sizes and types, but that depends on the Genes/tests
            that call this cx'er, or else you could build a Neuron that asplodes at runtime
        '''
        self.input = tf.Variable(initial_value=input_initial_value, dtype=PRECISION)
        self.feedback = tf.Variable(initial_value=feedback_initial_value, dtype=PRECISION)
        self.output = tf.Variable(initial_value=output_initial_value, dtype=PRECISION)

        self.input_hidden = tf.convert_to_tensor(input_hidden, dtype=PRECISION)
        self.hidden_feedback = tf.convert_to_tensor(hidden_feedback, dtype=PRECISION)
        self.feedback_hidden = tf.convert_to_tensor(feedback_hidden, dtype=PRECISION)
        self.hidden_output = tf.convert_to_tensor(hidden_output, dtype=PRECISION)
        # maya was here
        # *headbutt* - cece
        # boys rule girls drool - edgar
        self.activation = activation
        self.precision = precision
        self.input_ports = {}

    def update(self) -> None:
        new_val: ndarray = self.input.numpy()
        for port, input_cell in self.input_ports.items():
            new_val[port % len(new_val)] = input_cell.get_output()
        self.input.assign(new_val)
        hidden_in = tf.expand_dims(tf.concat([self.input, self.feedback], 0), 0)
        hidden_wts = tf.concat([self.input_hidden, self.feedback_hidden], 0)
        hidden = self.activation(tf.matmul(hidden_in, hidden_wts))
        self.feedback.assign(tf.squeeze(self.activation(tf.matmul(hidden, self.hidden_feedback)), 0))
        self.output.assign(tf.squeeze(self.activation(tf.matmul(hidden, self.hidden_output)), [0]))

    def get_output(self) -> PRECISION:
        return self.output.numpy()

    def add_input_connection(self, tx_cell: Cell, req_port: int):
        num_ports = self.input.shape[0]
        for offset in range(num_ports):
            rx_port = (req_port + offset) % num_ports
            if rx_port not in self.input_ports:
                self.input_ports[rx_port] = tx_cell
                return



class InputCell(Cell):

    def __init__(self, initial_value: PRECISION = None):
        self.value = initial_value

    def set_output(self, value: PRECISION):
        self.value = value

    def get_output(self) -> PRECISION:
        return self.value
