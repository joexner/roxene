import unittest

import numpy as np
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import ResizeNeuronLayer, LayerToResize
from roxene.util import set_rng

SEED = 111


class ResizeNeuronLayerMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    def test_resize_hidden_layer(self):
        """Test that ResizeNeuronLayer randomly widens or narrows the hidden layer"""
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        original_hidden_size = original_gene.input_hidden.shape[1]
        
        mutagen = ResizeNeuronLayer(LayerToResize.HIDDEN)
        
        # Run multiple times to see both widen and narrow
        widened = False
        narrowed = False
        for seed in range(100):
            set_rng(default_rng(seed))
            mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
            new_hidden_size = mutant_gene.input_hidden.shape[1]
            
            if new_hidden_size > original_hidden_size:
                widened = True
            elif new_hidden_size < original_hidden_size:
                narrowed = True
            
            if widened and narrowed:
                break
        
        self.assertTrue(widened, "Should see widening in 100 trials")
        self.assertTrue(narrowed, "Should see narrowing in 100 trials")

    def test_resize_preserves_shapes(self):
        """Test that all weight matrices have consistent shapes after resizing"""
        input_size = 8
        feedback_size = 6
        hidden_size = 10
        original_gene = CreateNeuron(**random_neuron_state(input_size, feedback_size, hidden_size))
        
        mutagen = ResizeNeuronLayer(LayerToResize.HIDDEN)
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        
        new_hidden_size = mutant_gene.input_hidden.shape[1]
        
        # Check all shapes are consistent
        self.assertEqual(mutant_gene.input_hidden.shape, (input_size, new_hidden_size))
        self.assertEqual(mutant_gene.feedback_hidden.shape, (feedback_size, new_hidden_size))
        self.assertEqual(mutant_gene.hidden_feedback.shape, (new_hidden_size, feedback_size))
        self.assertEqual(mutant_gene.hidden_output.shape, (new_hidden_size, 1))
        
        # Initial values should be unchanged in size
        self.assertEqual(mutant_gene.input.shape, (input_size,))
        self.assertEqual(mutant_gene.feedback.shape, (feedback_size,))
        self.assertEqual(mutant_gene.output.shape, (1,))

    def test_narrow_layer_minimum_size(self):
        """Test that narrowing preserves at least 1 hidden neuron"""
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 1))
        
        mutagen = ResizeNeuronLayer(LayerToResize.HIDDEN)
        
        # Try narrowing multiple times - layer of size 1 should never go below 1
        for seed in range(20):
            set_rng(default_rng(seed))
            mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
            self.assertGreaterEqual(mutant_gene.input_hidden.shape[1], 1)

    def test_persist_reload(self):
        """Test that ResizeNeuronLayer can be persisted and reloaded"""
        mutagen = ResizeNeuronLayer(LayerToResize.HIDDEN, 0.03)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(ResizeNeuronLayer, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.layer, LayerToResize.HIDDEN)
            self.assertEqual(reloaded.base_susceptibility, 0.03)
