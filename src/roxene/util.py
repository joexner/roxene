import threading
from typing import Dict

from numpy import ndarray, sign, exp, log
from numpy.random import Generator

from .constants import NP_PRECISION

thread_local_data = threading.local()

def set_rng(rng: Generator) -> None:
    """
    Bind a seeded RNG to the calling thread.

    Deliberately does not patch uuid.uuid4. Identifiers have to be unique, not
    reproducible: deriving them from the deterministic per-pod seed meant every
    restart replayed the same ID stream, so new trials collided on their primary
    key. IDs now come from real entropy, while the seeded RNG still drives all
    of the evolutionary randomness.
    """
    thread_local_data.rng = rng

def get_rng() -> Generator:
    return thread_local_data.rng

def random_slice(shape, rng: Generator = None) -> ndarray:
    """Generate a random array with values in [-1, 1]."""
    rng = rng or get_rng()
    return (2 * rng.random(shape) - 1).astype(dtype=NP_PRECISION)


def random_neuron_state(input_size=10, feedback_size=10, hidden_size=10, rng: Generator = None) -> Dict[str, ndarray]:
    rng = rng or get_rng()
    return {
        "input": random_slice([input_size], rng),
        "feedback": random_slice([feedback_size], rng),
        "output": random_slice([1], rng),
        "input_hidden": random_slice([input_size, hidden_size], rng),
        "hidden_feedback": random_slice([hidden_size, feedback_size], rng),
        "feedback_hidden": random_slice([feedback_size, hidden_size], rng),
        "hidden_output": random_slice([hidden_size, 1], rng),
    }


def wiggle(x, log_wiggle, absolute_wiggle = None, rng: Generator = None):
    """
    Randomly vary a value x != 0 by
    y = e^ln(x +/- log_wiggle) +/- absolute_wiggle
    keeping the sign
    """
    rng = rng or get_rng()
    log_wiggled = sign(x) * exp(rng.normal(log(abs(x)), log_wiggle))
    return rng.normal(log_wiggled, absolute_wiggle) if absolute_wiggle else log_wiggled
