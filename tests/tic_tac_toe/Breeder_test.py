import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import Engine

from roxene.tic_tac_toe import Environment
from roxene.tic_tac_toe.environment import BreederState
from roxene.tic_tac_toe.__main__ import run_breeder
from util import get_engine


class Breeder_test(unittest.TestCase):
    """Tests for the dynamic-frequency gate in run_breeder.

    run_breeder decides whether to cull+breed based on how many trials have
    completed since the last cycle (stored in breeder_state). cull/breed and
    the completion count are stubbed so the tests exercise the gate's decision
    and state transitions in isolation.
    """

    def _make_env(self) -> Environment:
        # get_engine() drops + recreates every table, so call it exactly once
        # per test and reuse the returned engine for the whole test.
        engine: Engine = get_engine()
        return Environment(engine)

    def _seed_state(self, env: Environment, trials_completed: int) -> None:
        with env.sessionmaker.begin() as session:
            session.add(BreederState(
                id=1,
                trials_completed_at_last_breed=trials_completed,
                last_breed_at=datetime.now(),
            ))

    def _read_state(self, env: Environment) -> BreederState:
        with env.sessionmaker() as session:
            return session.get(BreederState, 1)

    def _args(self, breed_every_n_trials: int) -> SimpleNamespace:
        return SimpleNamespace(breed_every_n_trials=breed_every_n_trials)

    def _stub_cycle(self, env: Environment, completed: int) -> None:
        env.count_trials = mock.Mock(return_value=completed)
        env.cull = mock.Mock(return_value=True)
        env.breed = mock.Mock(return_value=True)

    def test_skips_when_below_threshold(self):
        env = self._make_env()
        self._seed_state(env, trials_completed=10)
        self._stub_cycle(env, completed=30)  # 30 - 10 = 20 < 50

        run_breeder(self._args(50), env, seed=0, metrics_port=0)

        env.cull.assert_not_called()
        env.breed.assert_not_called()
        self.assertEqual(10, self._read_state(env).trials_completed_at_last_breed)

    def test_breeds_when_threshold_reached(self):
        env = self._make_env()
        self._seed_state(env, trials_completed=10)
        self._stub_cycle(env, completed=60)  # 60 - 10 = 50 >= 50

        run_breeder(self._args(50), env, seed=0, metrics_port=0)

        env.cull.assert_called_once()
        env.breed.assert_called_once()
        # High-water mark advances to the completed total for the next run.
        self.assertEqual(60, self._read_state(env).trials_completed_at_last_breed)

    def test_first_run_seeds_baseline_and_skips(self):
        # No prior state: the counter is baselined on the current total, so the
        # very first tick never breeds on stale, pre-existing trials.
        env = self._make_env()
        self._stub_cycle(env, completed=42)

        run_breeder(self._args(50), env, seed=0, metrics_port=0)

        env.cull.assert_not_called()
        env.breed.assert_not_called()
        self.assertEqual(42, self._read_state(env).trials_completed_at_last_breed)

    def test_frequency_scales_with_completion_rate(self):
        # At a steady completion rate the cycle should fire exactly once per
        # --breed_every_n_trials new trials, regardless of how slowly they
        # arrive. 250 trials at 25/tick with a threshold of 50 -> 5 cycles.
        env = self._make_env()
        self._seed_state(env, trials_completed=0)
        env.cull = mock.Mock(return_value=True)
        env.breed = mock.Mock(return_value=True)

        threshold = 50
        for completed in range(25, 251, 25):
            env.count_trials = mock.Mock(return_value=completed)
            run_breeder(self._args(threshold), env, seed=0, metrics_port=0)

        self.assertEqual(250 // threshold, env.breed.call_count)
        self.assertEqual(250 // threshold, env.cull.call_count)
        self.assertEqual(250, self._read_state(env).trials_completed_at_last_breed)

    def test_skips_when_completion_count_regresses(self):
        # If the completed count ever drops below the stored baseline (e.g. a
        # wiped table), the gate clamps to zero new trials rather than breed
        # on a negative delta.
        env = self._make_env()
        self._seed_state(env, trials_completed=100)
        self._stub_cycle(env, completed=50)

        run_breeder(self._args(50), env, seed=0, metrics_port=0)

        env.cull.assert_not_called()
        env.breed.assert_not_called()
        self.assertEqual(100, self._read_state(env).trials_completed_at_last_breed)
