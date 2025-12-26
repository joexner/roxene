import unittest

import numpy as np
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import ModifyInitialValue, InitialValueType
from roxene.util import set_rng

SEED = 444


class ModifyInitialValueMutagen_test(unittest.TestCase):

    def test_modify_initial_value_input(self):
        """Test that ModifyInitialValue modifies input initial values"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyInitialValue(InitialValueType.input, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some values should be different
        self.assertFalse(np.array_equal(mutant_gene.input, original_gene.input))
        # Other values should be unchanged
        np.testing.assert_array_equal(mutant_gene.feedback, original_gene.feedback)
        np.testing.assert_array_equal(mutant_gene.output, original_gene.output)

    def test_modify_initial_value_feedback(self):
        """Test that ModifyInitialValue modifies feedback initial values"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyInitialValue(InitialValueType.feedback, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some values should be different
        self.assertFalse(np.array_equal(mutant_gene.feedback, original_gene.feedback))
        # Other values should be unchanged
        np.testing.assert_array_equal(mutant_gene.input, original_gene.input)
        np.testing.assert_array_equal(mutant_gene.output, original_gene.output)

    def test_modify_initial_value_output(self):
        """Test that ModifyInitialValue modifies output initial values"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyInitialValue(InitialValueType.output, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some values should be different
        self.assertFalse(np.array_equal(mutant_gene.output, original_gene.output))
        # Other values should be unchanged
        np.testing.assert_array_equal(mutant_gene.input, original_gene.input)
        np.testing.assert_array_equal(mutant_gene.feedback, original_gene.feedback)

    def test_modify_initial_value_susceptibility(self):
        """Test that susceptibility controls mutation rate"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(20, 20, 20))
        
        # Low susceptibility - few changes
        mutagen_low = ModifyInitialValue(InitialValueType.input, 0.01, 0)
        mutant_low = mutagen_low.mutate(original_gene)
        changes_low = np.sum(mutant_low.input != original_gene.input)
        
        # High susceptibility - many changes
        mutagen_high = ModifyInitialValue(InitialValueType.input, 0.5, 0)
        mutant_high = mutagen_high.mutate(original_gene)
        changes_high = np.sum(mutant_high.input != original_gene.input)
        
        # Higher susceptibility should cause more changes
        self.assertGreater(changes_high, changes_low)

    def test_modify_initial_value_preserves_weights(self):
        """Test that weight matrices are preserved when modifying initial values"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyInitialValue(InitialValueType.input, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # All weight matrices should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)

    def test_persist_reload(self):
        """Test that ModifyInitialValue can be persisted and reloaded"""
        mutagen = ModifyInitialValue(InitialValueType.feedback, 0.018, 0.028)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ModifyInitialValue, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.layer, InitialValueType.feedback)
            self.assertEqual(reloaded.base_susceptibility, 0.018)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.028)
