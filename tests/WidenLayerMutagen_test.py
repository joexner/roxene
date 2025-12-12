import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import WidenLayerMutagen
from roxene.util import set_rng

SEED = 111


class WidenLayerMutagen_test(unittest.TestCase):

    def test_widen_layer(self):
        """Test that WidenLayerMutagen increases hidden layer size"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        original_hidden_size = original_gene.input_hidden.shape[1]
        
        mutagen = WidenLayerMutagen(1.0, 0)  # 100% susceptibility
        
        mutant_gene = mutagen.mutate(original_gene)
        new_hidden_size = mutant_gene.input_hidden.shape[1]
        
        # Hidden size should increase by 1-3
        self.assertGreater(new_hidden_size, original_hidden_size)
        self.assertLessEqual(new_hidden_size, original_hidden_size + 3)

    def test_widen_layer_preserves_shapes(self):
        """Test that all weight matrices have correct shapes after widening"""
        set_rng(default_rng(SEED))
        input_size = 8
        feedback_size = 6
        hidden_size = 10
        original_gene = CreateNeuron(**random_neuron_state(input_size, feedback_size, hidden_size))
        
        mutagen = WidenLayerMutagen(1.0, 0)  # 100% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        new_hidden_size = mutant_gene.input_hidden.shape[1]
        
        # Check all shapes are consistent
        self.assertEqual(mutant_gene.input_hidden.shape, (input_size, new_hidden_size))
        self.assertEqual(mutant_gene.feedback_hidden.shape, (feedback_size, new_hidden_size))
        self.assertEqual(mutant_gene.hidden_feedback.shape, (new_hidden_size, feedback_size))
        self.assertEqual(mutant_gene.hidden_output.shape, (new_hidden_size, 1))
        
        # Initial values should be unchanged
        self.assertEqual(mutant_gene.input.shape, (input_size,))
        self.assertEqual(mutant_gene.feedback.shape, (feedback_size,))
        self.assertEqual(mutant_gene.output.shape, (1,))

    def test_widen_layer_preserves_existing_weights(self):
        """Test that existing weights are preserved when widening"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        original_hidden_size = original_gene.input_hidden.shape[1]
        
        mutagen = WidenLayerMutagen(1.0, 0)  # 100% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        # Check that original weights are preserved in the first columns/rows
        import numpy as np
        np.testing.assert_array_equal(
            mutant_gene.input_hidden[:, :original_hidden_size],
            original_gene.input_hidden
        )
        np.testing.assert_array_equal(
            mutant_gene.feedback_hidden[:, :original_hidden_size],
            original_gene.feedback_hidden
        )
        np.testing.assert_array_equal(
            mutant_gene.hidden_feedback[:original_hidden_size, :],
            original_gene.hidden_feedback
        )
        np.testing.assert_array_equal(
            mutant_gene.hidden_output[:original_hidden_size, :],
            original_gene.hidden_output
        )

    def test_widen_layer_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = WidenLayerMutagen(0.0, 0)  # 0% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_persist_reload(self):
        """Test that WidenLayerMutagen can be persisted and reloaded"""
        mutagen = WidenLayerMutagen(0.03, 0.05)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(WidenLayerMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.03)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.05)
