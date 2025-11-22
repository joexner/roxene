import unittest
import numpy as np
from numpy.random import default_rng
from roxene import Mutagen, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import CreateNeuronMutagen, CNLayer

class Mutagen_test(unittest.TestCase):

    def test_susceptibility_caching(self):
        rng = default_rng(42)
        base = 0.01
        mutagen = CreateNeuronMutagen(CNLayer.input_hidden, base, 0.01)
        gene = CreateNeuron(**random_neuron_state(10, 10, 10, rng))
        val1 = mutagen.get_mutation_susceptibility(gene, rng)
        val2 = mutagen.get_mutation_susceptibility(gene, rng)
        self.assertEqual(val1, val2)
        self.assertIn(gene, mutagen.susceptibilities)

    def test_parent_susceptibility_inheritance(self):
        rng = default_rng(7)
        mutagen = CreateNeuronMutagen(CNLayer.input_hidden, 0.05, 0.1)
        parent = CreateNeuron(**random_neuron_state(10, 10, 10, rng))
        child = CreateNeuron(**random_neuron_state(10, 10, 10, rng), parent_gene=parent)
        parent_val = mutagen.get_mutation_susceptibility(parent, rng)
        child_val = mutagen.get_mutation_susceptibility(child, rng)
        self.assertNotEqual(parent_val, child_val)
        self.assertIn(parent, mutagen.susceptibilities)
        self.assertIn(child, mutagen.susceptibilities)

