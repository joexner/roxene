import unittest

from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import NarrowLayerMutagen
from roxene.util import set_rng

SEED = 222


class NarrowLayerMutagen_test(unittest.TestCase):

    def test_narrow_layer(self):
        """Test that NarrowLayerMutagen decreases hidden layer size"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        original_hidden_size = original_gene.input_hidden.shape[1]
        
        mutagen = NarrowLayerMutagen(1.0, 0)  # 100% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        new_hidden_size = mutant_gene.input_hidden.shape[1]
        
        # Hidden size should decrease by 1-2
        self.assertLess(new_hidden_size, original_hidden_size)
        self.assertGreaterEqual(new_hidden_size, original_hidden_size - 2)

    def test_narrow_layer_preserves_shapes(self):
        """Test that all weight matrices have correct shapes after narrowing"""
        set_rng(default_rng(SEED))
        input_size = 8
        feedback_size = 6
        hidden_size = 10
        original_gene = CreateNeuron(**random_neuron_state(input_size, feedback_size, hidden_size))
        
        mutagen = NarrowLayerMutagen(1.0, 0)  # 100% susceptibility
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

    def test_narrow_layer_minimum_size(self):
        """Test that narrowing preserves at least 1 hidden neuron"""
        set_rng(default_rng(SEED))
        # Create a gene with small hidden layer
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 2))
        
        mutagen = NarrowLayerMutagen(1.0, 0)  # 100% susceptibility
        
        # Try narrowing multiple times
        for _ in range(10):
            mutant_gene = mutagen.mutate(original_gene)
            # Layers with size <= 2 won't be narrowed, ensuring at least 1 neuron remains
            self.assertGreaterEqual(mutant_gene.input_hidden.shape[1], 1)

    def test_narrow_layer_no_mutation_small_layer(self):
        """Test that very small layers are not narrowed"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 2))
        
        mutagen = NarrowLayerMutagen(1.0, 0)  # 100% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not narrow a layer of size 2
        self.assertEqual(mutant_gene.input_hidden.shape[1], 2)

    def test_narrow_layer_no_mutation(self):
        """Test that with 0% susceptibility, no mutation occurs"""
        set_rng(default_rng(SEED))
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        
        mutagen = NarrowLayerMutagen(0.0, 0)  # 0% susceptibility
        mutant_gene = mutagen.mutate(original_gene)
        
        # Should not be mutated
        self.assertEqual(mutant_gene, original_gene)

    def test_persist_reload(self):
        """Test that NarrowLayerMutagen can be persisted and reloaded"""
        mutagen = NarrowLayerMutagen(0.02, 0.04)
        mutagen_id = mutagen.id
        engine = create_engine("sqlite://")
        EntityBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(mutagen)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(NarrowLayerMutagen, mutagen_id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.id, mutagen_id)
            self.assertEqual(reloaded.base_susceptibility, 0.02)
            self.assertEqual(reloaded.susceptibility_log_wiggle, 0.04)
