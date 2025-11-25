import unittest
import numpy as np
from numpy.random import default_rng

from roxene import Mutagen, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import CreateNeuronMutagen, CNLayer

class Mutagen_test(unittest.TestCase):

    def test_parent_susceptibility_inheritance(self):
        rng = default_rng(7)
        mutagen = CreateNeuronMutagen(CNLayer.input_hidden, 0.05, 0.1)

        other_mutagen = CreateNeuronMutagen(CNLayer.hidden_feedback, 0.05, 0.1)

        grandparent = CreateNeuron(**random_neuron_state(10, 10, 10, rng))
        parent = other_mutagen.mutate_CreateNeuron(grandparent, rng)
        child = other_mutagen.mutate_CreateNeuron(parent, rng)

        # Test susceptibility inheritance and caching across three generations
        self.assertNotIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(parent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        # Check that fetching the sus for the parent gets the grandparent too, but not the child
        parent_val = mutagen.get_mutation_susceptibility(parent, rng)
        self.assertIn(grandparent, mutagen.susceptibilities)
        self.assertIn(parent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        #Fetching the child populates its entry in the cache
        child_val = mutagen.get_mutation_susceptibility(child, rng)
        self.assertIn(child, mutagen.susceptibilities)

        grandparent_val = mutagen.get_mutation_susceptibility(grandparent, rng)

        self.assertNotEqual(grandparent_val, parent_val)
        self.assertNotEqual(parent_val, child_val)

