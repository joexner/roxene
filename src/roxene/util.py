from typing import Dict

from numpy import ndarray, sign, exp, log
from numpy.random import Generator, default_rng

from .constants import NP_PRECISION


def random_neuron_state(input_size=10, feedback_size=10, hidden_size=10, rng: Generator = None) -> Dict[str, ndarray]:
    rng = rng or default_rng()
    return {
        "input": (2 * rng.random([input_size]) - 1).astype(dtype=NP_PRECISION),
        "feedback": (2 * rng.random([feedback_size]) - 1).astype(dtype=NP_PRECISION),
        "output": (2 * rng.random([1]) - 1).astype(dtype=NP_PRECISION),
        "input_hidden": (2 * rng.random([input_size, hidden_size]) - 1).astype(dtype=NP_PRECISION),
        "hidden_feedback": (2 * rng.random([hidden_size, feedback_size]) - 1).astype(dtype=NP_PRECISION),
        "feedback_hidden": (2 * rng.random([feedback_size, hidden_size]) - 1).astype(dtype=NP_PRECISION),
        "hidden_output": (2 * rng.random([hidden_size, 1]) - 1).astype(dtype=NP_PRECISION),
    }


def wiggle(x, rng: Generator, log_wiggle, absolute_wiggle=0):
    """
    Randomly vary a value x != 0 by
    y = e^log(x +/- log_wiggle) +/- absolute_wiggle
    keeping the sign
    """
    log_wiggled = sign(x) * exp(rng.normal(log(abs(x)), log_wiggle))
    return rng.normal(log_wiggled, absolute_wiggle)
