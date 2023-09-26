import unittest

import numpy as np
import tensorflow as tf
from numpy.random import default_rng
# // maya smells...fine
from parameterized import parameterized

from cells import InputCell, Neuron
from util import random_neuron_state

SEED = 732478534

activation = tf.nn.tanh
precision = tf.dtypes.float16


class Neuron_test(unittest.TestCase):

    def setUp(self) -> None:
        tf.compat.v1.set_random_seed(SEED)

    @parameterized.expand([
        (2, 2, 4),
        (2, 3, 8),
        (1, 9, 1),
        (50, 9, 2),
    ])
    def test_update_updates(
            self,
            input_sz,
            hidden_sz,
            feedback_sz):

        neuron = Neuron(**random_neuron_state(input_sz, hidden_sz, feedback_sz))
        output_before_update = neuron.get_output()

        neuron.update()
        output_after_first_update = neuron.get_output()
        self.assertNotEqual(output_before_update, output_after_first_update)

        neuron.update()
        output_after_another_update = neuron.get_output()
        self.assertNotEqual(output_after_first_update, output_after_another_update)

    def test_check_math_linear(self):
        neuron: Neuron = Neuron(**random_neuron_state(1, 1, 1))

        initial_input_val = neuron.input.numpy()[0]
        initial_feedback_val = neuron.feedback.numpy()[0]

        input_hidden_weight = neuron.input_hidden.numpy()[0][0]
        feedback_hidden_weight = neuron.feedback_hidden.numpy()[0][0]

        neuron.update()

        expected_hidden_val = np.tanh(initial_input_val * input_hidden_weight + initial_feedback_val * feedback_hidden_weight)

        hidden_feedback_weight = neuron.hidden_feedback.numpy()[0][0]
        expected_feedback_val = np.tanh(expected_hidden_val * hidden_feedback_weight)
        actual_feedback_val = neuron.feedback.numpy()[0]
        self.assertAlmostEqual(actual_feedback_val, expected_feedback_val)

        hidden_out_weight = neuron.hidden_output.numpy()[0][0]
        expected_output_val = np.tanh(expected_hidden_val * hidden_out_weight)
        actual_output_val = neuron.get_output()
        self.assertAlmostEqual(actual_output_val, expected_output_val)

    @parameterized.expand([
        (2, 2, 4),
        (2, 3, 8),
        (1, 9, 1),
        (50, 9, 2),
    ])
    def test_input(
            self,
            input_sz,
            hidden_sz,
            feedback_sz,
    ):

        rng = default_rng(seed=SEED)

        initial_value = {
            "input": rng.uniform(low=-1., high=1., size=(input_sz)),
            "feedback": rng.uniform(low=-1., high=1., size=(feedback_sz)),
            "output": rng.uniform(low=-1., high=1., size=(1)),
            "input_hidden": rng.uniform(low=-1., high=1., size=(input_sz, hidden_sz)),
            "hidden_feedback": rng.uniform(low=-1., high=1., size=(hidden_sz, feedback_sz)),
            "feedback_hidden": rng.uniform(low=-1., high=1., size=(feedback_sz, hidden_sz)),
            "hidden_output": rng.uniform(low=-1., high=1., size=(hidden_sz, 1)),
        }

        neuron = Neuron(**initial_value)

        # Connect some input ports
        num_to_connect = rng.integers(1, input_sz) if input_sz > 1 else 1
        ports_to_connect = rng.choice(input_sz, num_to_connect, replace=False)
        connected_ports = {}
        for port in ports_to_connect:
            new_input_cell = InputCell(np.float16(rng.random()))
            neuron.add_input_connection(new_input_cell, port)
            connected_ports[port] = new_input_cell

        neuron.update()

        neuron_input_value: np.ndarray = neuron.input.numpy()
        for port_num in range(0, neuron_input_value.shape[0]):
            if port_num in connected_ports.keys():
                input_cell = connected_ports[port_num]
                expected_value = input_cell.get_output()
            else:
                expected_value = initial_value["input"][port_num]
            actual_value = neuron_input_value.flat[port_num]
            self.assertAlmostEqual(expected_value, actual_value, 3)


