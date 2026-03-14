import unittest

import numpy as np
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron, CompositeGene
from roxene.mutagens import WiggleCreateNeuron, CNLayer
from roxene.util import set_rng

SEED = 333


class WiggleCreateNeuron_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_wiggle_input_hidden_weights(self):
        """
        Test that WiggleCreateNeuron mutates the input_hidden weight matrix.
        
        When configured to target the input_hidden layer, the mutagen should
        modify those weights while leaving all other layers unchanged.
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden)
        
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        
        # The mutated gene should be a CreateNeuron with modified input_hidden weights
        self.assertIsInstance(mutant_gene, CreateNeuron)
        
        # input_hidden should be different (wiggled)
        self.assertFalse(
            np.allclose(mutant_gene.input_hidden, original_gene.input_hidden),
            "input_hidden weights should be modified"
        )
        
        # Other layers should be unchanged
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)
        np.testing.assert_array_equal(mutant_gene.input, original_gene.input)
        np.testing.assert_array_equal(mutant_gene.feedback, original_gene.feedback)
        np.testing.assert_array_equal(mutant_gene.output, original_gene.output)

    def test_wiggle_hidden_output_weights(self):
        """
        Test that WiggleCreateNeuron mutates the hidden_output weight matrix.
        
        The hidden_output layer connects the hidden neurons to the output,
        and is critical for how the neuron produces its final output.
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WiggleCreateNeuron(CNLayer.hidden_output)
        
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        
        # hidden_output should be different
        self.assertFalse(
            np.allclose(mutant_gene.hidden_output, original_gene.hidden_output),
            "hidden_output weights should be modified"
        )
        
        # Other weight matrices should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)

    def test_wiggle_input_initial_value(self):
        """
        Test that WiggleCreateNeuron mutates the input initial value vector.
        
        Initial values determine the starting state of neurons before
        any inputs are processed.
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WiggleCreateNeuron(CNLayer.input_initial_value)
        
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        
        # input initial values should be different
        self.assertFalse(
            np.allclose(mutant_gene.input, original_gene.input),
            "input initial values should be modified"
        )
        
        # Weight matrices should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)

    def test_severity_affects_wiggle_magnitude(self):
        """
        Test that higher severity values result in larger changes to weights.
        
        Severity controls the magnitude of the wiggle transformation:
        - Low severity should make small adjustments
        - High severity should make larger adjustments
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        # Low severity mutagen
        mutagen_low = WiggleCreateNeuron(CNLayer.input_hidden, 0.1)
        
        # High severity mutagen (max value is 1.0)
        mutagen_high = WiggleCreateNeuron(CNLayer.input_hidden, 1.0)
        
        set_rng(default_rng(SEED))
        mutant_low = mutagen_low.mutate_CreateNeuron(original_gene)
        
        set_rng(default_rng(SEED))
        mutant_high = mutagen_high.mutate_CreateNeuron(original_gene)
        
        # Calculate the change magnitude for each
        diff_low = np.abs(mutant_low.input_hidden - original_gene.input_hidden).mean()
        diff_high = np.abs(mutant_high.input_hidden - original_gene.input_hidden).mean()
        
        # High severity should cause larger changes on average
        self.assertGreater(diff_high, diff_low, 
            "Higher severity should cause larger weight changes")

    def test_parent_gene_set(self):
        """
        Test that the mutated gene properly references the original as parent_gene.
        
        This maintains the gene lineage for tracking evolutionary history.
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden)
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        
        self.assertEqual(mutant_gene.parent_gene, original_gene)

    def test_no_mutation_at_zero_susceptibility(self):
        """
        Test that WiggleCreateNeuron doesn't mutate at 0% susceptibility.
        """
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WiggleCreateNeuron(CNLayer.input_hidden, 0.0)  # 0% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should be the exact same object (no mutation)
        self.assertIs(mutant_gene, original_gene)

    def test_persist_reload(self):
        """
        Test that WiggleCreateNeuron can be persisted and reloaded with all
        configuration preserved.
        """
        mutagen = WiggleCreateNeuron(CNLayer.hidden_feedback, base_susceptibility=0.02)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        
        with Session(engine) as session:
            reloaded = session.get(WiggleCreateNeuron, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.02)
            self.assertEqual(reloaded.layer, CNLayer.hidden_feedback)
            self.assertEqual(reloaded.severity, 1.0)