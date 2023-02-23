import unittest

import numpy as np
from numpy import ndarray
from numpy.random import Generator

from roxene import CreateNeuron
from roxene.genes import CNLayer
from roxene.mutagens import CreateNeuronMutagen
from roxene.util import random_neuron_state

SEED = 2837457


class Mutagens_test(unittest.TestCase):

    def test_CreateNeuronMutagen(self):
        rng: Generator = np.random.default_rng(SEED)
        gene = CreateNeuron(**random_neuron_state(1000, 1000, 1000))
        for layer_to_mutate in CNLayer:
            mutagen = CreateNeuronMutagen(layer_to_mutate=layer_to_mutate)
            mutant = mutagen.mutate(gene, rng)
            self.check_fraction_changed(layer_to_mutate is CNLayer.input_initial_value,
                                        mutant.input_initial_value, gene.input_initial_value)
            self.check_fraction_changed(layer_to_mutate is CNLayer.input_hidden, mutant.input_hidden,
                                        gene.input_hidden)
            self.check_fraction_changed(layer_to_mutate is CNLayer.hidden_feedback, mutant.hidden_feedback,
                                        gene.hidden_feedback)
            self.check_fraction_changed(layer_to_mutate is CNLayer.feedback_initial_value,
                                        mutant.feedback_initial_value, gene.feedback_initial_value)
            self.check_fraction_changed(layer_to_mutate is CNLayer.feedback_hidden, mutant.feedback_hidden,
                                        gene.feedback_hidden)
            self.check_fraction_changed(layer_to_mutate is CNLayer.hidden_output, mutant.hidden_output,
                                        gene.hidden_output)
            self.check_fraction_changed(layer_to_mutate is CNLayer.output_initial_value,
                                        mutant.output_initial_value, gene.output_initial_value)

    def check_fraction_changed(self, expect_any_change: bool, original: ndarray, mutant: ndarray):
        num_changed = 0
        num_unchanged = 0
        for o_ex, m_ex in zip(np.nditer(original), np.nditer(mutant)):
            if o_ex == m_ex:
                num_unchanged += 1
            else:
                num_changed += 1

        fraction_changed = float(num_changed) / float(num_changed + num_unchanged)

        if expect_any_change:
            self.assertGreaterEqual(fraction_changed, .5, "More than 50% of the values should have changed")
            if (num_changed + num_unchanged) > 100:
                self.assertLessEqual(fraction_changed, 1.,
                                     "At least of the values should have been unchanged, due to, umm, fp16 precision")
        else:
            self.assertEqual(fraction_changed, 0, "None should have changed")
