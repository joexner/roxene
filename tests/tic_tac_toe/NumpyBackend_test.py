import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng

from roxene import Organism, Neuron
from roxene.util import random_neuron_state, set_rng
from roxene.tic_tac_toe import Environment
from util import get_engine

SEED = 11235


def make_organism(rng):
    organism = Organism(input_names={"A", "B"}, output_names={"O1", "O2"})
    n1 = Neuron(**random_neuron_state(1, 1, 1, rng=rng))
    n2 = Neuron(**random_neuron_state(1, 1, 1, rng=rng))
    organism.addNeuron(n1)
    organism.addNeuron(n2)
    n1.add_input_connection(n2, 0)
    n2.add_input_connection(n1, 0)
    return organism, n1, n2


class NumpyBackend_test(unittest.TestCase):

    def setUp(self):
        set_rng(default_rng(seed=SEED))

    def _numpy_patch(self):
        return (mock.patch("roxene.organism.NUMPY_BACKEND", "on"),
                mock.patch("roxene.organism.NUMPY_BACKEND_MAX_FLOPS", 10 ** 9))

    def test_default_backend_is_torch(self):
        organism, n1, n2 = make_organism(default_rng(SEED))
        organism.update()
        self.assertNotEqual("numpy", organism._compiled_execution.get("backend"))

    def test_update_compiles_numpy_backend(self):
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            before = [n1.get_output(), n2.get_output()]
            organism.update()

            compiled = organism._compiled_execution
            self.assertEqual("numpy", compiled["backend"])
            self.assertIsInstance(compiled["C"], np.ndarray)
            self.assertEqual(np.float32, compiled["C"].dtype)
            self.assertIsInstance(n1.output, np.ndarray)
            self.assertIsInstance(n1.input, np.ndarray)
            # state moved, outputs changed
            self.assertNotEqual(before[0], n1.get_output())
            self.assertNotEqual(before[1], n2.get_output())

    def test_set_input_and_sync_on_numpy_state(self):
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            organism.update()
            organism.set_input("A", 0.5)
            # wrote into the numpy C
            self.assertTrue(np.isfinite(organism._compiled_execution["C"]).all())
            organism.sync_neuron_inputs()
            # neuron input views see the routed input
            self.assertTrue(np.isfinite(n1.input).all())

    def test_db_roundtrip_with_numpy_state(self):
        engine = get_engine()
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            organism.update()
            out_name = next(name for name, n in organism.outputs.items() if n is n1)
            saved_output = float(n1.get_output())

            from sqlalchemy.orm import Session
            with Session(engine) as session:
                session.add(organism)
                session.commit()
                oid = organism.id
            with Session(engine) as session:
                reloaded = session.get(Organism, oid)
                outputs = {k: float(v.get_output()) for k, v in reloaded.outputs.items()}

            self.assertAlmostEqual(saved_output, outputs[out_name], places=5)

    def test_full_trial_with_numpy_backend(self):
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            env = Environment(get_engine())
            env.populate(2)
            trial = env.start_trial()
            trial.run(timeout=100)
            env.complete_trial(trial)
            self.assertGreater(len(trial.moves), 0)


if __name__ == "__main__":
    unittest.main()