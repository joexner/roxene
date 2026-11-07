import unittest

import numpy as np
from numpy.random import default_rng
# // maya smells...fine
from parameterized import parameterized
from sqlalchemy.orm import Session

from roxene import InputCell, Neuron, Organism, random_neuron_state
from roxene.util import set_rng
from tic_tac_toe.util import get_engine

SEED = 732478534


class Neuron_test(unittest.TestCase):

    def setUp(self) -> None:
        set_rng(default_rng(SEED))

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

        neuron = Neuron(**random_neuron_state(input_sz, hidden_sz, feedback_sz, rng=default_rng(SEED)))
        org = Organism()
        org.addNeuron(neuron)
        output_before_update = neuron.get_output()

        org.update()
        org.commit_state()
        output_after_first_update = neuron.get_output()
        self.assertFalse(np.array_equal(output_before_update, output_after_first_update))

        org.update()
        org.commit_state()
        output_after_another_update = neuron.get_output()
        self.assertFalse(np.array_equal(output_after_first_update, output_after_another_update))

    def test_check_math_linear(self):
        neuron: Neuron = Neuron(**random_neuron_state(1, 1, 1, rng=default_rng(SEED)))
        org = Organism()
        org.addNeuron(neuron)

        # dynamics run in fp16 (TORCH_PRECISION), so the reference is computed at that precision
        initial_input_val = np.float16(neuron.input[0])
        initial_feedback_val = np.float16(neuron.feedback[0])

        input_hidden_weight = np.float16(neuron.input_hidden[0][0])
        feedback_hidden_weight = np.float16(neuron.feedback_hidden[0][0])

        org.update()
        org.commit_state()

        expected_hidden_val = np.tanh(initial_input_val * input_hidden_weight + initial_feedback_val * feedback_hidden_weight)

        hidden_feedback_weight = np.float16(neuron.hidden_feedback[0][0])
        expected_feedback_val = np.tanh(expected_hidden_val * hidden_feedback_weight)
        actual_feedback_val = neuron.feedback[0]
        self.assertAlmostEqual(actual_feedback_val, expected_feedback_val)

        hidden_out_weight = np.float16(neuron.hidden_output[0][0])
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
        org = Organism()
        org.addNeuron(neuron)

        # Connect some input ports
        num_to_connect = rng.integers(1, input_sz) if input_sz > 1 else 1
        ports_to_connect = rng.choice(input_sz, num_to_connect, replace=False)
        connected_ports = {}
        for port in ports_to_connect:
            new_input_cell = InputCell(np.float16(rng.random()))
            neuron.add_input_connection(new_input_cell, port)
            org.cells.append(new_input_cell)
            connected_ports[port] = new_input_cell

        org.update()
        org.commit_state()

        neuron_input_value: np.ndarray = neuron.input
        for port_num in range(0, neuron_input_value.shape[0]):
            if port_num in connected_ports.keys():
                input_cell = connected_ports[port_num]
                expected_value = input_cell.get_output()
            else:
                expected_value = initial_value["input"][port_num]
            actual_value = neuron_input_value.flat[port_num]
            self.assertAlmostEqual(expected_value, actual_value, 3)

    def test_save_neuron(self):
        engine = get_engine()

        n1 = Neuron(**random_neuron_state())
        o1 = Organism()
        o1.addNeuron(n1)
        nid = n1.id
        oid = o1.id

        with Session(engine) as session:
            session.add(o1)
            orig_input = n1.input.copy()
            orig_feedback = n1.feedback.copy()
            orig_output = n1.output.copy()
            orig_input_hidden = n1.input_hidden.copy()
            orig_hidden_feedback = n1.hidden_feedback.copy()
            orig_feedback_hidden = n1.feedback_hidden.copy()
            orig_hidden_output = n1.hidden_output.copy()
            session.commit()

        with Session(engine) as session:
            o2 = session.get(Organism, oid)
            n2 = session.get(Neuron, nid)

            self.assertFalse(n2 is None)

            np.testing.assert_array_equal(n2.input, orig_input)
            np.testing.assert_array_equal(n2.feedback, orig_feedback)
            np.testing.assert_array_equal(n2.output, orig_output)
            np.testing.assert_array_equal(n2.input_hidden, orig_input_hidden)
            np.testing.assert_array_equal(n2.hidden_feedback, orig_hidden_feedback)
            np.testing.assert_array_equal(n2.feedback_hidden, orig_feedback_hidden)
            np.testing.assert_array_equal(n2.hidden_output, orig_hidden_output)

            o2.update()
            o2.commit_state()

            n2_input = n2.input.copy()
            n2_feedback = n2.feedback.copy()
            n2_output = n2.output.copy()
            np.testing.assert_array_equal(n2_input, orig_input) # input is not changed w/ update()
            self.assertFalse(np.allclose(n2_feedback, orig_feedback))
            self.assertNotEqual(float(n2_output[0]), float(orig_output[0]))
            session.commit()

        with Session(engine) as session:
            n3 = session.get(Neuron, nid)
            n3_input = n3.input.copy()
            n3_feedback = n3.feedback.copy()
            n3_output = n3.output.copy()
            self.assertNotEqual(float(n3_output[0]), float(orig_output[0]))
            np.testing.assert_array_equal(n3_input, n2_input)
            np.testing.assert_array_equal(n3_feedback, n2_feedback)
            np.testing.assert_array_equal(n3_output, n2_output)

    def test_save_linked_neurons(self):
        engine = get_engine()

        n1 = Neuron(**random_neuron_state())
        n2 = Neuron(**random_neuron_state())
        n3 = Neuron(**random_neuron_state())
        n4 = Neuron(**random_neuron_state())

        n1.add_input_connection(n2, 0)
        n1.add_input_connection(n3, 1)
        n1.add_input_connection(n4, 2)

        n1_id = n1.id
        n2_id = n2.id
        n3_id = n3.id
        n4_id = n4.id

        with Session(engine) as session:
            session.add(n1)
            session.add(n2)
            session.add(n3)
            session.add(n4)
            session.commit()

        with Session(engine) as session:
            n5 = session.get(Neuron, n1_id)
            self.assertFalse(n5 is None)
            self.assertEqual(len(n5.bound_ports), 3)
            self.assertEqual(n5.bound_ports[0].id, n2_id)
            self.assertEqual(n5.bound_ports[1].id, n3_id)
            self.assertEqual(n5.bound_ports[2].id, n4_id)
