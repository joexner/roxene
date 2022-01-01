import tensorflow as tf
from numpy.core.multiarray import ndarray

from Cell import Cell


class Neuron(Cell):
    PRECISION = tf.dtypes.float16
    ACTIVATION = tf.nn.tanh

    def __init__(self,
                 input_initial_value: tf.Tensor,
                 feedback_initial_value: tf.Tensor,
                 output_initial_value: tf.Tensor,
                 input_hidden: tf.Tensor,
                 hidden_feedback: tf.Tensor,
                 feedback_hidden: tf.Tensor,
                 hidden_output: tf.Tensor,
                 activation=ACTIVATION,
                 precision=PRECISION):
        self.input = tf.Variable(initial_value=input_initial_value)
        self.feedback = tf.Variable(initial_value=feedback_initial_value)
        self.output = tf.Variable(initial_value=output_initial_value, dtype=precision)

        self.input_hidden = input_hidden
        self.hidden_feedback = hidden_feedback
        self.feedback_hidden = feedback_hidden
        self.hidden_output = hidden_output
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
        self.output.assign(tf.squeeze(self.activation(tf.matmul(hidden, self.hidden_output)), [0, 1]))

    def get_output(self) -> PRECISION:
        return self.output.numpy()

    def add_input_connection(self, tx_cell: Cell, req_port: int):
        num_ports = self.input.shape[0]
        for offset in range(num_ports):
            rx_port = (req_port + offset) % num_ports
            if rx_port not in self.input_ports:
                self.input_ports[rx_port] = tx_cell
                return
