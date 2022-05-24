from unittest import TestCase
from unittest.mock import Mock

from tests.util import build_CN_params
from roxene import Organism, Gene, CompositeGene, CreateNeuron

SEED = 22049456


class Organism_test(TestCase):

    def test_constructor_genoytpe(self):
        root_gene = Mock(Gene)
        organism = Organism(genotype=[root_gene])
        print(root_gene.method_calls)
        root_gene.execute.assert_called_once_with(organism)

    def test_constructor_input_output_names(self):
        input_names = {'I_0', 'I_1', 'I_2'}
        output_names = {'O1', '02'}
        gene = CreateNeuron(**build_CN_params(input_size=2, feedback_size=3, hidden_size=1))
        organism = Organism(input_names, output_names, [CompositeGene(genes=[gene], iterations=10)])
        self.assertSetEqual(input_names, set(organism.inputs.keys()))
        self.assertSetEqual(output_names, set(organism.outputs.keys()))
