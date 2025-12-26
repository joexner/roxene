import unittest

import numpy as np
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import ModifyWeight, WeightLayer
from roxene.util import set_rng

SEED = 333


class ModifyWeightMutagen_test(unittest.TestCase):

    def test_modify_weight_input_hidden(self):
        """Test that ModifyWeight modifies input_hidden weights"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyWeight(WeightLayer.input_hidden, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some weights should be different
        self.assertFalse(np.array_equal(mutant_gene.input_hidden, original_gene.input_hidden))
        # Other weights should be unchanged
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)

    def test_modify_weight_hidden_feedback(self):
        """Test that ModifyWeight modifies hidden_feedback weights"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyWeight(WeightLayer.hidden_feedback, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some weights should be different
        self.assertFalse(np.array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback))
        # Other weights should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)

    def test_modify_weight_feedback_hidden(self):
        """Test that ModifyWeight modifies feedback_hidden weights"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyWeight(WeightLayer.feedback_hidden, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some weights should be different
        self.assertFalse(np.array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden))
        # Other weights should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)
        np.testing.assert_array_equal(mutant_gene.hidden_output, original_gene.hidden_output)

    def test_modify_weight_hidden_output(self):
        """Test that ModifyWeight modifies hidden_output weights"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(10, 10, 10))
        
        mutagen = ModifyWeight(WeightLayer.hidden_output, 0.5, 0)
        mutant_gene = mutagen.mutate(original_gene)
        
        # Some weights should be different
        self.assertFalse(np.array_equal(mutant_gene.hidden_output, original_gene.hidden_output))
        # Other weights should be unchanged
        np.testing.assert_array_equal(mutant_gene.input_hidden, original_gene.input_hidden)
        np.testing.assert_array_equal(mutant_gene.hidden_feedback, original_gene.hidden_feedback)
        np.testing.assert_array_equal(mutant_gene.feedback_hidden, original_gene.feedback_hidden)

    def test_modify_weight_susceptibility(self):
        """Test that susceptibility controls mutation rate"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(20, 20, 20))
        
        # Low susceptibility - few changes
        mutagen_low = ModifyWeight(WeightLayer.input_hidden, 0.01, 0)
        mutant_low = mutagen_low.mutate(original_gene)
        changes_low = np.sum(mutant_low.input_hidden != original_gene.input_hidden)
        
        # High susceptibility - many changes
        mutagen_high = ModifyWeight(WeightLayer.input_hidden, 0.5, 0)
        mutant_high = mutagen_high.mutate(original_gene)
        changes_high = np.sum(mutant_high.input_hidden != original_gene.input_hidden)
        
        # Higher susceptibility should cause more changes
        self.assertGreater(changes_high, changes_low)

    def test_persist_reload(self):
        """Test that ModifyWeight can be persisted and reloaded"""
        mutagen = ModifyWeight(WeightLayer.input_hidden, 0.025, 0.035)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ModifyWeight, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.layer, WeightLayer.input_hidden)
            self.assertEqual(reloaded.base_susceptibility, 0.025)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.035)
