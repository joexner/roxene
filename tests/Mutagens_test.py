import unittest

import numpy as np
from numpy import ndarray
from numpy.random import Generator

from roxene import CreateNeuron
from roxene.mutagens import NeuronInitialValueMutagen
from roxene.util import random_neuron_state

SEED = 2837457


class Mutagens_test(unittest.TestCase):

    def test_NeuronInitialValueMutagen(self):
        rng: Generator = np.random.default_rng(SEED)
        gene = CreateNeuron(**random_neuron_state(500, 500, 500))
        mutagen = NeuronInitialValueMutagen()
        mutant = mutagen.mutate(gene, rng)

        self.assertSomeChanged(mutant.input_initial_value, gene.input_initial_value)
        self.assertSomeChanged(mutant.input_hidden, gene.input_hidden)
        self.assertSomeChanged(mutant.hidden_feedback, gene.hidden_feedback)
        self.assertSomeChanged(mutant.feedback_initial_value, gene.feedback_initial_value)
        self.assertSomeChanged(mutant.feedback_hidden, gene.feedback_hidden)
        self.assertSomeChanged(mutant.hidden_output, gene.hidden_output)
        self.assertSomeChanged(mutant.output_initial_value, gene.output_initial_value)

    def assertSomeChanged(self, original: ndarray, mutant: ndarray):
        num_changed = 0.
        num_unchanged = 0.
        for o_ex, m_ex in zip(np.nditer(original), np.nditer(mutant)):
            if o_ex == m_ex:
                num_unchanged += 1.
            else:
                num_changed += 1.

        fraction_changed = num_changed / (num_changed + num_unchanged)

        self.assertGreaterEqual(fraction_changed, .5, "More than 50% of the values should have changed")
        if (num_changed + num_unchanged) > 100:
            self.assertLessEqual(fraction_changed, 1., "At least of the values should have been unchanged")
