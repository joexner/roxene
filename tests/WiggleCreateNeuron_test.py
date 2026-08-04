import unittest

import numpy as np
from numpy.random import default_rng
from parameterized import parameterized
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from roxene import EntityBase, random_neuron_state
from roxene.genes import CreateNeuron
from roxene.mutagens import WiggleCreateNeuron, CNLayer
from roxene.util import set_rng

SEED = 11235

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
        """Check that the target layer is wiggled and the others are not"""
        gene = CreateNeuron(**random_neuron_state(5, 5, 10))
        mutagen = WiggleCreateNeuron(layer)
        mutant = mutagen.mutate_CreateNeuron(gene)
        self.assertIsInstance(mutant, CreateNeuron)
        self.assertIs(mutant.parent_gene, gene)
        for (_, attr_to_check) in _LAYER_CASES:
            if attr_to_check == attr_to_wiggle:
                self.assertFalse(np.any(getattr(mutant, attr_to_check) == getattr(gene, attr_to_check)))
            else:
                np.testing.assert_array_equal(getattr(mutant, attr_to_check), getattr(gene, attr_to_check))

    def test_severity_affects_wiggle_magnitude(self):
        """Check that severity affects wiggle magnitude"""
        original = CreateNeuron(**random_neuron_state(5, 5, 10))
        severities = [0.001, 0.01, 0.1, 0.5, 1.0]
        diffs = []

        for s in severities:
            set_rng(default_rng(SEED))
            mutagen = WiggleCreateNeuron(CNLayer.input_hidden, s)
            mutant = mutagen.mutate_CreateNeuron(original)
            diffs.append(np.abs(mutant.input_hidden - original.input_hidden).mean())

        for i in range(1, len(diffs)):
            msg = f"Expected wiggle magnitude to increase with severity, got {dict(zip(severities, diffs))}"
            self.assertGreater(diffs[i], diffs[i - 1], msg)

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