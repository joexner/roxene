"""Prometheus metrics for the roxene training stack.

The worker and breeder are long-running processes, so each can host a small
``/metrics`` HTTP endpoint (backed by ``prometheus_client``) for Prometheus to
scrape. Counters and the trial-duration histogram are updated inline as work
happens; the gauges that require a database query (pool size, in-flight and
total trials) are refreshed on a background thread so scrapes stay fast and
don't add query load to Postgres.

Everything is opt-in: nothing starts until :func:`start_server` is called with a
non-zero port, so notebooks and the unit-test suite are completely unaffected.
"""
from __future__ import annotations

import collections
import logging
import threading
from typing import TYPE_CHECKING

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

if TYPE_CHECKING:
    from .environment import Environment

logger = logging.getLogger(__name__)

# --- Updated inline as work happens (no DB access, cheap) ---
TRIALS_COMPLETED = Counter(
    "roxene_trials_completed_total",
    "Trials finished by this process.",
)
TRIAL_DURATION = Histogram(
    "roxene_trial_duration_seconds",
    "Wall-clock time to run a single trial.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
MOVES_TOTAL = Counter(
    "roxene_moves_total",
    "Total moves played by this process.",
)
OUTCOMES_TOTAL = Counter(
    "roxene_outcomes_total",
    "Moves tallied by outcome (WIN/LOSE/TIE/TIMEOUT/VALID_MOVE/INVALID_MOVE).",
    ["outcome"],
)
ORGANISMS_BRED = Counter(
    "roxene_organisms_bred_total",
    "Organisms bred by this process.",
)
ORGANISMS_CULLED = Counter(
    "roxene_organisms_culled_total",
    "Organisms culled by this process.",
)

# --- DB-backed gauges, refreshed on a timer (see MetricsServer) ---
POOL_SIZE = Gauge(
    "roxene_pool_size",
    "Living organisms currently in the pool.",
)
TRIALS_RUNNING = Gauge(
    "roxene_trials_running",
    "Trials currently in flight (started but not finished).",
)
TRIALS_TOTAL = Gauge(
    "roxene_trials_total",
    "Total trials recorded so far.",
)


def record_trial(trial, duration_seconds: float) -> None:
    """Record one finished trial's duration, move count, and outcome breakdown."""
    TRIALS_COMPLETED.inc()
    TRIAL_DURATION.observe(duration_seconds)
    # NB: fully qualified collections.Counter -- the bare name `Counter` in this
    # module is prometheus_client's Counter (used for the metrics above).
    outcome_counts: collections.Counter[str] = collections.Counter()
    for move in trial.moves:
        for outcome in move.outcomes:
            outcome_counts[outcome.name] += 1
    MOVES_TOTAL.inc(len(trial.moves))
    for name, count in outcome_counts.items():
        OUTCOMES_TOTAL.labels(outcome=name).inc(count)


def record_breeds(num: int) -> None:
    ORGANISMS_BRED.inc(num)


def record_culls(num: int) -> None:
    ORGANISMS_CULLED.inc(num)


class MetricsServer:
    """Runs the /metrics HTTP server plus a gauge refresher in daemon threads."""

    def __init__(self, port: int, env: Environment | None = None, refresh_seconds: float = 15.0):
        self._env = env
        self._refresh_seconds = refresh_seconds
        self._stop = threading.Event()
        # Bind 0.0.0.0 so the endpoint is reachable from other pods, not just localhost.
        start_http_server(port, addr="0.0.0.0")
        if env is not None:
            threading.Thread(target=self._refresh_loop, name="roxene-metrics-refresher", daemon=True).start()
        logger.info(f"Metrics endpoint listening on :{port}")

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            try:
                POOL_SIZE.set(self._env.count_organisms())
                TRIALS_RUNNING.set(self._env.count_trials(started=True, completed=False))
                TRIALS_TOTAL.set(self._env.count_trials())
            except Exception:  # a failing gauge scrape must never take down the worker/breeder
                logger.exception("Failed to refresh roxene metric gauges")
            self._stop.wait(self._refresh_seconds)

    def stop(self) -> None:
        self._stop.set()


def start_server(port: int | None, env: Environment | None = None) -> MetricsServer | None:
    """Start the metrics server if ``port`` is set and non-zero, else no-op."""
    if not port:
        return None
    return MetricsServer(int(port), env)
