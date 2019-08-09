import unittest
import tensorflow as tf
# // maya smells...fine
from Perceptron import Perceptron

SEED = 732478534

activation = tf.nn.tanh
precision = tf.dtypes.float16


class TestPerceptron(unittest.TestCase):

    def setUp(self) -> None:
        tf.enable_eager_execution()
        tf.set_random_seed(SEED)


    def test_update_does_not_explode(self):
        input_sz = 3
        hidden_sz = 7
        feedback_sz = 5

        perceptron = Perceptron(
            input_initial_value=random_tensor([input_sz]),
            feedback_initial_value=random_tensor([feedback_sz]),
            output_initial_value=random_tensor([]),

            input_hidden=random_tensor([input_sz, hidden_sz]),
            hidden_feedback=random_tensor([hidden_sz, feedback_sz]),
            feedback_hidden=random_tensor([feedback_sz, hidden_sz]),
            hidden_output=random_tensor([hidden_sz, 1])
        )

        perceptron.update()
        print(perceptron.get_output())

    def test_random_seed(self):
        input_initial_value=random_tensor([1])
        print(input_initial_value.numpy())

    def test_random_seed_again(self):
        input_initial_value=random_tensor([1])
        print(input_initial_value.numpy())

    def test_eager_execution(self):
        x = [[2.]]
        m = tf.matmul(x, x)
        print("hello, {}".format(m))


def random_tensor(size):
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=precision)


def identity(x: tf.float16):
    return x
