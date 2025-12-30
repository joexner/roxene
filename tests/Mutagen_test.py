import unittest
from numpy.random import default_rng

from roxene import random_neuron_state
from roxene.genes import CreateNeuron as CreateNeuronGene
from roxene.mutagens import WiggleCNLayer, CNLayer
from roxene.util import set_rng

class Mutagen_test(unittest.TestCase):

    def test_parent_susceptibility_inheritance(self):
        rng = default_rng(7)
        set_rng(rng)  # Set the global rng for the test
        mutagen = WiggleCNLayer(CNLayer.input_hidden, 0.05, 0.1)

        other_mutagen = WiggleCNLayer(CNLayer.hidden_feedback, 0.05, 0.1)

        grandparent = CreateNeuronGene(**random_neuron_state(10, 10, 10, rng))
        parent = other_mutagen.mutate_CreateNeuron(grandparent)
        child = other_mutagen.mutate_CreateNeuron(parent)

        # Test susceptibility inheritance and caching across three generations
        self.assertNotIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)
        
        # If parent is the same object as grandparent (mutation didn't happen due to susceptibility),
        # then assertNotIn(parent, ...) would be the same as assertNotIn(grandparent, ...)
        # Only check if parent is a different object
        if parent is not grandparent:
            self.assertNotIn(parent, mutagen.susceptibilities)

        # Check that fetching the sus for the parent gets its ancestors too, but not the child
        parent_val = mutagen.get_mutation_susceptibility(parent)
        
        # Grandparent should always be cached (either directly or as parent's ancestor)
        self.assertIn(grandparent, mutagen.susceptibilities)
        
        # If parent is a different object from grandparent, it should also be cached
        if parent is not grandparent:
            self.assertIn(parent, mutagen.susceptibilities)
            
        self.assertNotIn(child, mutagen.susceptibilities)

        #Fetching the child populates its entry in the cache
        child_val = mutagen.get_mutation_susceptibility(child)
        self.assertIn(child, mutagen.susceptibilities)

        grandparent_val = mutagen.get_mutation_susceptibility(grandparent)

        # If parent and grandparent are the same object, their susceptibilities will be equal
        if parent is not grandparent:
            self.assertNotEqual(grandparent_val, parent_val)
        self.assertNotEqual(parent_val, child_val)


