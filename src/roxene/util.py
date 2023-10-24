from typing import Dict

from numpy import ndarray
from numpy.random import Generator, default_rng

from .constants import NP_PRECISION


def random_neuron_state(input_size, feedback_size, hidden_size, rng: Generator = default_rng()) -> Dict[str, ndarray]:
    return {
        "input_initial_value": (2 * rng.random([input_size]) - 1).astype(dtype=NP_PRECISION),
        "feedback_initial_value": (2 * rng.random([feedback_size]) - 1).astype(dtype=NP_PRECISION),
        "output_initial_value": (2 * rng.random([1]) - 1).astype(dtype=NP_PRECISION),
        "input_hidden": (2 * rng.random([input_size, hidden_size]) - 1).astype(dtype=NP_PRECISION),
        "hidden_feedback": (2 * rng.random([hidden_size, feedback_size]) - 1).astype(dtype=NP_PRECISION),
        "feedback_hidden": (2 * rng.random([feedback_size, hidden_size]) - 1).astype(dtype=NP_PRECISION),
        "hidden_output": (2 * rng.random([hidden_size, 1]) - 1).astype(dtype=NP_PRECISION),
    }
