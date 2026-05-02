import unittest
from parameterized import parameterized

import numpy as np
from numpy import array_equal
from numpy.testing import assert_array_equal
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import ResizeNeuronLayer, LayerToResize
from roxene.mutagens.resize_neuron_layer import ResizeDirection
from roxene.util import set_rng

SEED = 11235

ALL_ATTRS = {"input", "feedback", "output", "input_hidden", "hidden_feedback", "feedback_hidden", "hidden_output"}

SIZE_ATTR = {
    LayerToResize.INPUT: "input",
    LayerToResize.HIDDEN: "hidden_output",
    LayerToResize.FEEDBACK: "feedback",
}

RESIZED_ATTRS = {
    LayerToResize.INPUT: {"input", "input_hidden"},
    LayerToResize.HIDDEN: {"input_hidden", "hidden_feedback", "feedback_hidden", "hidden_output"},
    LayerToResize.FEEDBACK: {"feedback", "hidden_feedback", "feedback_hidden"},
}


class ResizeNeuronLayerMutagen_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    @parameterized.expand([
        (LayerToResize.INPUT, ResizeDirection.WIDEN,),
        (LayerToResize.INPUT, ResizeDirection.NARROW,),
        (LayerToResize.HIDDEN, ResizeDirection.WIDEN,),
        (LayerToResize.HIDDEN, ResizeDirection.NARROW,),
        (LayerToResize.FEEDBACK, ResizeDirection.WIDEN,),
        (LayerToResize.FEEDBACK, ResizeDirection.NARROW,),
    ])
    def test_changes_layer_sizes(self, layer_to_resize, direction):
        """Test that ResizeNeuronLayer can widen and narrow layers"""
        mutagen = ResizeNeuronLayer(layer_to_resize, direction)
        original_gene = CreateNeuron(**random_neuron_state())
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        original_size = len(getattr(original_gene, SIZE_ATTR[layer_to_resize]))
        mutant_size = len(getattr(mutant_gene, SIZE_ATTR[layer_to_resize]))
        if direction == ResizeDirection.WIDEN:
            self.assertEqual(mutant_size, original_size + 1, f"{layer_to_resize.name} layer should widen by 1")
        else:
            self.assertEqual(mutant_size, original_size - 1, f"{layer_to_resize.name} layer should narrow  by 1")


    @parameterized.expand([
        (LayerToResize.INPUT,),
        (LayerToResize.HIDDEN,),
        (LayerToResize.FEEDBACK,),
    ])
    def test_narrow_minimum_size(self, layer_to_resize):
        """Test that each layer type preserves minimum size of 1"""
        mutagen = ResizeNeuronLayer(layer_to_resize, ResizeDirection.NARROW)
        original_gene = CreateNeuron(**random_neuron_state(
            input_size=1 if layer_to_resize == LayerToResize.INPUT else 5,
            feedback_size=1 if layer_to_resize == LayerToResize.FEEDBACK else 5,
            hidden_size=1 if layer_to_resize == LayerToResize.HIDDEN else 5 ))
        
        set_rng(default_rng(42))
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        size_attr = SIZE_ATTR[layer_to_resize]
        new_size = len(getattr(mutant_gene, size_attr))
        self.assertGreaterEqual(new_size, 1, f"{layer_to_resize.name} layer should never go below 1")


    @parameterized.expand([
        (LayerToResize.INPUT, ResizeDirection.WIDEN,),
        (LayerToResize.INPUT, ResizeDirection.NARROW,),
        (LayerToResize.HIDDEN, ResizeDirection.WIDEN,),
        (LayerToResize.HIDDEN, ResizeDirection.NARROW,),
        (LayerToResize.FEEDBACK, ResizeDirection.WIDEN,),
        (LayerToResize.FEEDBACK, ResizeDirection.NARROW,),
    ])
    def test_value_preservation(self, layer_to_resize, direction):
        """Test that non-resized arrays maintain original values and resized arrays have new values"""
        mutagen = ResizeNeuronLayer(layer_to_resize, direction)
        original_gene = CreateNeuron(**random_neuron_state())
        mutant_gene = mutagen.mutate_CreateNeuron(original_gene)
        for attr in ALL_ATTRS:
            orig_val = getattr(original_gene, attr)
            mut_val = getattr(mutant_gene, attr)
            if attr not in RESIZED_ATTRS[layer_to_resize]:
                assert_array_equal(mut_val, orig_val, f"{attr} should be unchanged when resizing {layer_to_resize.name}")
            else:
                self.assertFalse(array_equal(mut_val, orig_val), f"{attr} should be changed when resizing")
                larger, smaller = (mut_val, orig_val) if direction == ResizeDirection.WIDEN else (orig_val, mut_val)
                for ax in range(larger.ndim):
                    if larger.shape[ax] == smaller.shape[ax] + 1:
                        resized_ax = ax
                        break
                self.assertIsNotNone(resized_ax, f"{attr}: should have a resized dimension")
                found = False
                for removed_idx in range(larger.shape[resized_ax]):
                    if array_equal(np.delete(larger, removed_idx, axis=resized_ax), smaller):
                        found = True
                        break
                self.assertTrue(found, f"{attr}: should recover original by removing one slice along dimension {resized_ax}")

    @parameterized.expand([
        (LayerToResize.INPUT, ResizeDirection.WIDEN,),
        (LayerToResize.HIDDEN, ResizeDirection.WIDEN,),
        (LayerToResize.FEEDBACK, ResizeDirection.WIDEN,),
        (LayerToResize.INPUT, ResizeDirection.NARROW,),
        (LayerToResize.HIDDEN, ResizeDirection.NARROW,),
        (LayerToResize.FEEDBACK, ResizeDirection.NARROW,),
    ])
    def test_persist_reload_all_layer_types(self, layer_type, direction):
        """Test that ResizeNeuronLayer can be persisted and reloaded for all layer types"""
        mutagen = ResizeNeuronLayer(layer_type, direction, 0.03)
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
            self.assertEqual(reloaded.layer, layer_type)
            self.assertEqual(reloaded.direction, direction)
            self.assertEqual(reloaded.base_susceptibility, 0.03)
