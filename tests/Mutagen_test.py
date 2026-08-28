import unittest
from unittest.mock import patch

from numpy.random import default_rng

from roxene import Mutagen, random_neuron_state
from roxene.genes import CreateNeuron as CreateNeuronGene, ConnectNeurons, CompositeGene, RotateCells
from roxene.mutagens import WiggleCreateNeuron, CNLayer
from roxene.util import set_rng

SEED = 11235


class Mutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_susceptibility(self):
        gene = CreateNeuronGene(**random_neuron_state())

        # Test base_susceptibility=0.0: no mutation should occur
        low_sus = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=0.0)
        result = low_sus.mutate(gene)
        self.assertIs(result, gene)

        # Test base_susceptibility=1.0: mutation should occur
        hi_sus = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=1.0)
        result = hi_sus.mutate(gene)
        self.assertIsNot(result, gene)

    def test_susceptibility_inheritance(self):
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden)

        other_mutagen = WiggleCreateNeuron(CNLayer.hidden_feedback)
        grandparent = CreateNeuronGene(**random_neuron_state())
        parent = other_mutagen.mutate_CreateNeuron(grandparent)
        child = other_mutagen.mutate_CreateNeuron(parent)

        # Susceptibility hasn't been fetched yet for any of these genes yet, so they're not in the cache
        self.assertNotIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(parent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        # Check that fetching the sus for the parent gets its ancestors too, but not the child
        parent_val = mutagen.get_mutation_susceptibility(parent)
        self.assertIn(parent, mutagen.susceptibilities)
        self.assertIn(grandparent, mutagen.susceptibilities)
        self.assertNotIn(child, mutagen.susceptibilities)

        # Check that fetching the grandparent value populates it in the cache
        grandparent_val = mutagen.get_mutation_susceptibility(grandparent)
        self.assertEqual(grandparent_val, mutagen.base_susceptibility)

        # Fetching the child populates its entry in the cache
        child_val = mutagen.get_mutation_susceptibility(child)
        self.assertIn(child, mutagen.susceptibilities)

        # Make sure the grandparent gets the mutagen's base susceptibility, and
        # the sus values are different, because they were wiggled
        self.assertNotEqual(grandparent_val, parent_val)
        self.assertNotEqual(parent_val, child_val)

    def test_susceptibility_and_severity_validation(self):
        # Invalid base_susceptibility values
        with self.assertRaises(ValueError):
            WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=-0.1)
        with self.assertRaises(ValueError):
            WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=1.1)

        # Invalid severity values
        with self.assertRaises(ValueError):
            WiggleCreateNeuron(CNLayer.input_hidden, severity=-4.1)
        with self.assertRaises(ValueError):
            WiggleCreateNeuron(CNLayer.input_hidden, severity=100)

        # Valid boundary values should work
        mutagen_min = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=0.0, severity=0.0)
        self.assertEqual(mutagen_min.base_susceptibility, 0.0)
        self.assertEqual(mutagen_min.severity, 0.0)

        mutagen_max = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=1.0, severity=1.0)
        self.assertEqual(mutagen_max.base_susceptibility, 1.0)
        self.assertEqual(mutagen_max.severity, 1.0)

    def test_mutate_dispatches_by_gene_type(self):
        mutagen = Mutagen(base_susceptibility=1.0)
        cases = [
            (CompositeGene(), "mutate_CompositeGene"),
            (CreateNeuronGene(**random_neuron_state()), "mutate_CreateNeuron"),
            (ConnectNeurons(tx_cell_index=0, rx_input_port=0), "mutate_ConnectNeurons"),
            (RotateCells(), "mutate_RotateCells"),
        ]

        for gene, method_name in cases:
            with self.subTest(gene_type=type(gene).__name__):
                method = getattr(mutagen, method_name)
                with patch.object(Mutagen, method_name, wraps=method) as mock_method:
                    result = mutagen.mutate(gene)

                mock_method.assert_called_once_with(gene)
                self.assertIs(result, gene)

    def test_mutate_CompositeGene_recursive(self):
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=1.0)

        child1 = CreateNeuronGene(**random_neuron_state())
        child2 = CreateNeuronGene(**random_neuron_state())
        cg = CompositeGene([child1, child2], iterations=3)

        result = mutagen.mutate(cg)

        # New CompositeGene created with mutated children
        self.assertIsNot(result, cg)
        self.assertIsInstance(result, CompositeGene)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(len(result.child_genes), 2)
        self.assertIsNot(result.child_genes[0], child1)
        self.assertIsNot(result.child_genes[1], child2)
        self.assertEqual(result.parent_gene, cg)

    def test_mutate_CompositeGene_nested(self):
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden, base_susceptibility=1.0)

        inner_child = CreateNeuronGene(**random_neuron_state())
        inner_cg = CompositeGene([inner_child], iterations=5)

        outer_connect = ConnectNeurons(tx_cell_index=0, rx_input_port=0)
        outer_cg = CompositeGene([inner_cg, outer_connect], iterations=1)

        result = mutagen.mutate(outer_cg)

        # Outer should become new CompositeGene
        self.assertIsNot(result, outer_cg)
        self.assertIsInstance(result, CompositeGene)

        # Inner should be mutated (CreateNeuron child)
        mutated_inner = result.child_genes[0]
        self.assertIsInstance(mutated_inner, CompositeGene)
        self.assertEqual(mutated_inner.iterations, 5)
        self.assertEqual(mutated_inner.parent_gene, inner_cg)

