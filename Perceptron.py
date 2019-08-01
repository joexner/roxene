import tensorflow as tf
from tensorflow import Tensor


class Perceptron:

    def __init__(self,
                 input_initial_value: Tensor,
                 feedback_initial_value: Tensor,
                 output_initial_value: float,
                 input_hidden: Tensor,
                 hidden_feedback: Tensor,
                 feedback_hidden: Tensor,
                 hidden_output: Tensor,
                 activation=tf.nn.tanh,
                 precision=tf.dtypes.float16):
        self.input = tf.Variable(initial_value=input_initial_value)
        self.feedback = tf.Variable(initial_value=feedback_initial_value)
        self.output = tf.Variable(initial_value=output_initial_value, dtype=precision)

        self.input_hidden = input_hidden
        self.hidden_feedback = hidden_feedback
        self.feedback_hidden = feedback_hidden
        self.hidden_output = hidden_output

        self.activation = activation
        self.precision = precision

    def update(self):
        hidden_in = tf.expand_dims(tf.concat([self.input, self.feedback], 0), 0)
        hidden_wts = tf.concat([self.input_hidden, self.feedback_hidden], 0)
        hidden = self.activation(tf.matmul(hidden_in, hidden_wts))
        self.feedback.assign(tf.squeeze(self.activation(tf.matmul(hidden, self.hidden_feedback))))
        self.output.assign(tf.squeeze(self.activation(tf.matmul(hidden, self.hidden_output))))

    def get_output(self):
        return self.output
