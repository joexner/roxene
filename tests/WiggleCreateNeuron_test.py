import unittest

import numpy as np
from numpy.random import default_rng
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from parameterized import parameterized

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import WiggleCreateNeuron, CNLayer
from roxene.util import set_rng

SEED = 333

# (layer_to_mutate, mutated_attr_name)
_LAYER_CASES = [
    (CNLayer.input_initial_value,    "input"),
    (CNLayer.feedback_initial_value, "feedback"),
    (CNLayer.output_initial_value,   "output"),
    (CNLayer.input_hidden,           "input_hidden"),
    (CNLayer.hidden_feedback,        "hidden_feedback"),
    (CNLayer.feedback_hidden,        "feedback_hidden"),
    (CNLayer.hidden_output,          "hidden_output"),
]


class WiggleCreateNeuron_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(SEED))

    @parameterized.expand(_LAYER_CASES)
    def test_wiggle_layer(self, layer, attr_to_wiggle):
        gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        mutagen = WiggleCreateNeuron(layer)
        mutant = mutagen.mutate_CreateNeuron(gene)
        self.assertIsInstance(mutant, CreateNeuron)
        for (_, attr_to_check) in _LAYER_CASES:
            if attr_to_check == attr_to_wiggle:
                self.assertFalse(np.any(getattr(mutant, attr_to_check) == getattr(gene, attr_to_check)))
            else:
                np.testing.assert_array_equal(getattr(mutant, attr_to_check), getattr(gene, attr_to_check))

    def test_severity_affects_wiggle_magnitude(self):
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))

        mutagen_low = WiggleCreateNeuron(CNLayer.input_hidden, 0.1)
        mutagen_high = WiggleCreateNeuron(CNLayer.input_hidden, 1.0)

        set_rng(default_rng(SEED))
        mutant_low = mutagen_low.mutate_CreateNeuron(original_gene)
        set_rng(default_rng(SEED))
        mutant_high = mutagen_high.mutate_CreateNeuron(original_gene)

        diff_low = np.abs(mutant_low.input_hidden - original_gene.input_hidden).mean()
        diff_high = np.abs(mutant_high.input_hidden - original_gene.input_hidden).mean()

        self.assertGreater(diff_high, diff_low)

    def test_parent_gene_set(self):
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        mutant_gene = WiggleCreateNeuron(CNLayer.input_hidden).mutate_CreateNeuron(original_gene)
        self.assertEqual(mutant_gene.parent_gene, original_gene)

    def test_no_mutation_at_zero_susceptibility(self):
        original_gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        mutant_gene = WiggleCreateNeuron(CNLayer.input_hidden, 0.0).mutate(original_gene)
        self.assertIs(mutant_gene, original_gene)

    def test_persist_reload(self):
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