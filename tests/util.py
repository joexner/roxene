import tensorflow as tf

from roxene import Neuron


def build_CN_params(input_size, feedback_size, hidden_size):
    return {
        "input_initial_value": tf.random.uniform([input_size], -1., 1., dtype=Neuron.PRECISION),
        "feedback_initial_value": tf.random.uniform([feedback_size], -1., 1., dtype=Neuron.PRECISION),
        "output_initial_value": tf.random.uniform([1], -1., 1., dtype=Neuron.PRECISION),
        "input_hidden": tf.random.uniform([input_size, hidden_size], -1., 1., dtype=Neuron.PRECISION),
        "hidden_feedback": tf.random.uniform([hidden_size, feedback_size], -1., 1., dtype=Neuron.PRECISION),
        "feedback_hidden": tf.random.uniform([feedback_size, hidden_size], -1., 1., dtype=Neuron.PRECISION),
        "hidden_output": tf.random.uniform([hidden_size], -1., 1., dtype=Neuron.PRECISION),
    }
