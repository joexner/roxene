import unittest
import tensorflow as tf
from tensorflow import Tensor
# // maya smells...fine
from Perceptron import Perceptron
import numpy as np

SEED = 732478534

activation = tf.nn.tanh
precision = tf.dtypes.float16


class Perceptron_test(unittest.TestCase):

    def setUp(self) -> None:
        tf.enable_eager_execution()
        tf.set_random_seed(SEED)


    def test_update_updates(self):
        input_sz = 3
        hidden_sz = 7
        feedback_sz = 5

        input_initial_value = random_tensor([input_sz])
        feedback_initial_value = random_tensor([feedback_sz])
        output_initial_value = random_tensor([])
        input_hidden = random_tensor([input_sz, hidden_sz])
        hidden_feedback = random_tensor([hidden_sz, feedback_sz])
        feedback_hidden = random_tensor([feedback_sz, hidden_sz])
        hidden_output = random_tensor([hidden_sz, 1])

        perceptron = Perceptron(
            input_initial_value=input_initial_value,
            feedback_initial_value=feedback_initial_value,
            output_initial_value=output_initial_value,
            input_hidden=input_hidden,
            hidden_feedback=hidden_feedback,
            feedback_hidden=feedback_hidden,
            hidden_output=hidden_output
        )

        output_before_update = perceptron.get_output().numpy()
        self.assertEqual(output_before_update, output_initial_value.numpy())

        perceptron.update()
        output_after_first_update = perceptron.get_output().numpy()
        self.assertNotEqual(output_before_update, output_after_first_update)

        perceptron.update()
        output_after_another_update = perceptron.get_output().numpy()
        self.assertNotEqual(output_after_first_update, output_after_another_update)

    def test_check_math_linear(self):
        input_initial_value = random_tensor([1])
        feedback_initial_value = random_tensor([1])
        output_initial_value = random_tensor([])
        input_hidden = random_tensor([1, 1])
        hidden_feedback = random_tensor([1, 1])
        feedback_hidden = random_tensor([1, 1])
        hidden_output = random_tensor([1, 1])

        perceptron = Perceptron(
            input_initial_value=input_initial_value,
            feedback_initial_value=feedback_initial_value,
            output_initial_value=output_initial_value,
            input_hidden=input_hidden,
            hidden_feedback=hidden_feedback,
            feedback_hidden=feedback_hidden,
            hidden_output=hidden_output
        )

        perceptron.update()

        input_val = input_initial_value.numpy()[0]
        input_weight = input_hidden.numpy()[0][0]

        feedback_val = feedback_initial_value.numpy()[0]
        feedback_weight = feedback_hidden.numpy()[0][0]

        hidden_val = np.tanh(input_val * input_weight + feedback_val * feedback_weight)

        hidden_feedback_weight = hidden_feedback.numpy()[0][0]
        expected_feedback_val = np.tanh(hidden_val * hidden_feedback_weight)
        self.assertAlmostEqual(perceptron.feedback.numpy(), expected_feedback_val)

        hidden_out_weight = hidden_output.numpy()[0][0]
        expected_output_val = np.tanh(hidden_val * hidden_out_weight)
        self.assertAlmostEqual(perceptron.get_output().numpy(), expected_output_val)


def random_tensor(size) -> Tensor:
    return tf.random.uniform(
        shape=size,
        minval=-1,
        maxval=1,
        dtype=precision)
