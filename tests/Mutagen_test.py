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
        grandparent = CreateNeuron(**random_neuron_state(10, 10, 10, rng))
        # Create parent by mutating grandparent with a different mutagen
        mutagen2 = CreateNeuronMutagen(CNLayer.hidden_feedback, 0.05, 0.1)
        parent = mutagen2.mutate_CreateNeuron(grandparent, rng)
        self.assertEqual(parent.parent_gene, grandparent)
        # Create child by mutating parent with another mutagen
        mutagen3 = CreateNeuronMutagen(CNLayer.hidden_output, 0.05, 0.1)
        child = mutagen3.mutate_CreateNeuron(parent, rng)
        self.assertEqual(child.parent_gene, parent)
        # Test susceptibility inheritance across three generations
        grandparent_val = mutagen.get_mutation_susceptibility(grandparent, rng)
        parent_val = mutagen.get_mutation_susceptibility(parent, rng)
        child_val = mutagen.get_mutation_susceptibility(child, rng)
        self.assertNotEqual(grandparent_val, parent_val)
        self.assertNotEqual(parent_val, child_val)
        self.assertIn(grandparent, mutagen.susceptibilities)
        self.assertIn(parent, mutagen.susceptibilities)
        self.assertIn(child, mutagen.susceptibilities)

