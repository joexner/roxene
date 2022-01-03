import unittest
from random import Random

import numpy as np
import tensorflow as tf
# // maya smells...fine
from parameterized import parameterized

from roxene import InputCell
from roxene import Neuron

from . import random_tensor

SEED = 732478534

activation = tf.nn.tanh
precision = tf.dtypes.float16


class Neuron_test(unittest.TestCase):

    def setUp(self) -> None:
        tf.compat.v1.enable_eager_execution()
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

        neuron = make_neuron(input_sz, hidden_sz, feedback_sz)
        output_before_update = neuron.get_output()

        neuron.update()
        output_after_first_update = neuron.get_output()
        self.assertNotEqual(output_before_update, output_after_first_update)

        neuron.update()
        output_after_another_update = neuron.get_output()
        self.assertNotEqual(output_after_first_update, output_after_another_update)

    def test_check_math_linear(self):
        neuron: Neuron = make_neuron(1, 1, 1)

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

        rng = Random(SEED)

        input_initial_value = random_tensor([input_sz])
        feedback_initial_value = random_tensor([feedback_sz])
        output_initial_value = random_tensor([])
        input_hidden = random_tensor([input_sz, hidden_sz])
        hidden_feedback = random_tensor([hidden_sz, feedback_sz])
        feedback_hidden = random_tensor([feedback_sz, hidden_sz])
        hidden_output = random_tensor([hidden_sz, 1])

        neuron = Neuron(
            input_initial_value=input_initial_value,
            feedback_initial_value=feedback_initial_value,
            output_initial_value=output_initial_value,
            input_hidden=input_hidden,
            hidden_feedback=hidden_feedback,
            feedback_hidden=feedback_hidden,
            hidden_output=hidden_output
        )

        # Connect some input ports
        num_to_connect = rng.randint(0, input_sz)
        ports_to_connect = rng.sample(range(input_sz), num_to_connect)
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
                expected_value = input_initial_value.numpy()[port_num]
            actual_value = neuron_input_value.flat[port_num]
            self.assertEqual(expected_value, actual_value)

    # def test_add_input_connection(self, input_sz, genes, exepcted_connections):
    #     self.fail()

def make_neuron(input_sz,
                hidden_sz,
                feedback_sz):

    input_initial_value = random_tensor([input_sz])
    feedback_initial_value = random_tensor([feedback_sz])
    output_initial_value = random_tensor([])
    input_hidden = random_tensor([input_sz, hidden_sz])
    hidden_feedback = random_tensor([hidden_sz, feedback_sz])
    feedback_hidden = random_tensor([feedback_sz, hidden_sz])
    hidden_output = random_tensor([hidden_sz, 1])

    return Neuron(
        input_initial_value=input_initial_value,
        feedback_initial_value=feedback_initial_value,
        output_initial_value=output_initial_value,
        input_hidden=input_hidden,
        hidden_feedback=hidden_feedback,
        feedback_hidden=feedback_hidden,
        hidden_output=hidden_output
    )

