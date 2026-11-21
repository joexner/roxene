import json
import unittest

import numpy as np
import torch
from numpy import finfo
from numpy.random import default_rng

from roxene import random_neuron_state
from roxene.cells.neuron import Neuron
from roxene.constants import NP_PRECISION
from roxene.genes import CreateNeuron
from roxene.mutagens import WiggleCreateNeuron, CNLayer
from roxene.util import set_rng, wiggle, clamp_to_precision

SEED = 11235
F16_MAX = float(finfo(NP_PRECISION).max)
LAYER_NAMES = ["input", "feedback", "output", "input_hidden", "hidden_feedback", "feedback_hidden", "hidden_output"]


class Util_test(unittest.TestCase):

    def test_clamp_to_precision(self):
        a = np.array([-1e30, -F16_MAX * 2, -1.0, 0.0, 1.0, F16_MAX * 2, 1e30], dtype=np.float64)
        clamped = clamp_to_precision(a)
        self.assertEqual(clamped.dtype, NP_PRECISION)
        self.assertTrue(np.isfinite(clamped).all())
        self.assertLessEqual(np.abs(clamped).max(), F16_MAX)

    def test_wiggle_scalar_stays_within_precision_range(self):
        set_rng(default_rng(SEED))
        for _ in range(1000):
            y = wiggle(0.001, 0.03)
            self.assertTrue(np.isfinite(y))
            self.assertLessEqual(abs(y), F16_MAX)
