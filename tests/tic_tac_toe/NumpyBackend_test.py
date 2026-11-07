import unittest
from unittest import mock

import numpy as np
from numpy.random import default_rng

from roxene import Organism, Neuron
from roxene.compiled_network import CompiledNetwork, NumpyCompiledNetwork
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
        network = organism._compiled_network
        self.assertIsInstance(network, CompiledNetwork)
        self.assertNotIsInstance(network, NumpyCompiledNetwork)

    def test_over_the_flops_threshold_uses_torch(self):
        # numpy enabled, but the organism is bigger than the threshold -> torch
        with mock.patch("roxene.organism.NUMPY_BACKEND", "on"), \
                mock.patch("roxene.organism.NUMPY_BACKEND_MAX_FLOPS", 1):
            organism, n1, n2 = make_organism(default_rng(SEED))
            organism.update()
            network = organism._compiled_network
            self.assertIsInstance(network, CompiledNetwork)
            self.assertNotIsInstance(network, NumpyCompiledNetwork)

    def test_update_compiles_numpy_backend(self):
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            before = [n1.get_output(), n2.get_output()]
            organism.update()

            network = organism._compiled_network
            self.assertIsInstance(network, NumpyCompiledNetwork)
            self.assertIsInstance(network.C, np.ndarray)
            self.assertEqual(np.float16, network.C.dtype)
            # neuron state is plain numpy even without a network
            self.assertIsInstance(n1.output, np.ndarray)
            self.assertIsInstance(n1.input, np.ndarray)
            # live state moved; after committing, the plain arrays see it
            organism.commit_state()
            self.assertNotEqual(before[0], n1.get_output())
            self.assertNotEqual(before[1], n2.get_output())

    def test_set_input_and_sync_on_numpy_state(self):
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            organism.update()
            organism.set_input("A", 0.5)
            # wrote into the numpy C
            self.assertTrue(np.isfinite(organism._compiled_network.C).all())
            organism.sync_neuron_inputs()
            organism.commit_state()
            # plain input arrays see the routed values
            self.assertTrue(np.isfinite(n1.input).all())

    def test_db_roundtrip_with_numpy_state(self):
        engine = get_engine()
        with self._numpy_patch()[0], self._numpy_patch()[1]:
            organism, n1, n2 = make_organism(default_rng(SEED))
            organism.update()
            out_name = next(name for name, n in organism.outputs.items() if n is n1)
            # the before_flush hook commits the live state before it is persisted
            saved_output = float(organism.get_output(out_name))

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