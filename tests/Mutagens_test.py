import unittest

from roxene import CreateNeuron, Neuron
from roxene.mutagens import NeuronInitialValueMutagen


class Mutagens_test(unittest.TestCase):

    def test_Initial(self):
        parent_gene_data = Neuron.random_neuron_state(5, 5, 5)
        gene = CreateNeuron(**parent_gene_data)

        mutagen: NeuronInitialValueMutagen = NeuronInitialValueMutagen()
        mutant = mutagen.mutate(gene)

        for parentGene, murantGene in zip(parent_gene_data.input_initial_value, mutant.input_initial_value):
            self.assertNotEqual()
        parent_gene_data.feedback_initial_value
        parent_gene_data.output_initial_value

        parent_initial_values = {(k, v) for k, v in parent_gene_data.items()
                                 # if k.endswith("initial_value")
                                 }
        mutant_initial_values = {(k, v) for k, v in mutant.items() if k.endswith("initial_value")}
        self.assertEqual(parent_initial_values, mutant_initial_values)
        parent_gene_data.input_initial_value
        self.assertEqual(True, False)  # add assertion here
