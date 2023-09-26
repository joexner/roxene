from unittest import TestCase

from numpy.random import default_rng, Generator
from unittest.mock import Mock

from genes import Gene, CompositeGene, CreateNeuron
from organism import Organism
from util import random_neuron_state

SEED = 22049456


class Organism_test(TestCase):

    def test_constructor_genoytpe(self):
        root_gene = Mock(Gene)
        organism = Organism(genotype=root_gene)
        print(root_gene.method_calls)
        root_gene.execute.assert_called_once_with(organism)

    def test_constructor_input_output_names(self):
        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**random_neuron_state(input_size=2, feedback_size=3, hidden_size=1))
        organism = Organism(input_names, output_names, CompositeGene(genes=[gene], iterations=10))
        self.assertSetEqual(input_names, set(organism.inputs.keys()))
        self.assertSetEqual(output_names, set(organism.outputs.keys()))

    def test_io(self):

        rng: Generator = default_rng(SEED)

        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**random_neuron_state(rng=rng))
        organism = Organism(input_names, output_names, CompositeGene(genes=[gene], iterations=10))


        for input_name in input_names:
            organism.set_input(input_name, rng.random())
        for output_name in output_names:
            organism.get_output(output_name)

        self.assertEqual(input_names, set(organism.inputs.keys()))
        self.assertEqual(output_names, set(organism.outputs.keys()))

