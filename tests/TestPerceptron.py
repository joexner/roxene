import unittest
import tensorflow as tf
# // maya smells...fine
from Perceptron import Perceptron
import random

activation = tf.nn.tanh
precision = tf.dtypes.float16


class TestPerceptron(unittest.TestCase):

    def test_update_does_not_explode(self):
        input_sz = 3
        hidden_sz = 7
        feedback_sz = 5

        tf.set_random_seed(732478534)

        perceptron = Perceptron(
            input_initial_value=random_tensor([input_sz]),
            feedback_initial_value=random_tensor([feedback_sz]),
            output_initial_value=random.uniform(-1, 1),

            input_hidden=random_tensor([input_sz, hidden_sz]),
            hidden_feedback=random_tensor([hidden_sz, feedback_sz]),
            feedback_hidden=random_tensor([feedback_sz, hidden_sz]),
            hidden_output=random_tensor([hidden_sz, 1])
        )

        perceptron.update()
        output = perceptron.get_output()
        print(output)


def random_tensor(size):
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=precision)


def identity(x: tf.float16):
    return x
