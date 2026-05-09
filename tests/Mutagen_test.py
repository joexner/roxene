import unittest
from numpy.random import default_rng

from roxene import random_neuron_state
from roxene.genes import CreateNeuron as CreateNeuronGene, CompositeGene, ConnectNeurons, RotateCells
from roxene.mutagens import WiggleCreateNeuron, CNLayer, ShuffleGenes, ModifyIterations, RetargetConnectNeurons
from roxene.util import set_rng

SEED = 11235


class Mutagen_test(unittest.TestCase):

    def test_parent_susceptibility_inheritance(self):
        set_rng(default_rng(SEED))

        other_mutagen = WiggleCreateNeuron(CNLayer.hidden_feedback)

        grandparent = CreateNeuronGene(**random_neuron_state())
        parent = other_mutagen.mutate_CreateNeuron(grandparent)
        child = other_mutagen.mutate_CreateNeuron(parent)

        # Test susceptibility inheritance and caching across three generations
        # New Mutagen, hasn't decided sus'ty any of these genes yet
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden)
        self.assertNotIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(parent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        # Check that fetching the sus for the parent gets its ancestors too, but not the child
        parent_val = mutagen.get_mutation_susceptibility(parent)
        self.assertIn(parent, mutagen.susceptibilities)
        self.assertIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        # Check that the grandparent gets the mutagen's base susceptibility
        grandparent_val = mutagen.get_mutation_susceptibility(grandparent)
        self.assertEqual(grandparent_val, mutagen.base_susceptibility)

        # Fetching the child populates its entry in the cache
        child_val = mutagen.get_mutation_susceptibility(child)
        self.assertIn(child, mutagen.susceptibilities)

        # Make sure the sus values are different, because they were wiggled
        self.assertNotEqual(grandparent_val, parent_val)
        self.assertNotEqual(parent_val, child_val)


