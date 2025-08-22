from numpy import sign, exp, log
from numpy.random import Generator


def wiggle(x, rng: Generator, log_wiggle, absolute_wiggle=0):
    """
    Randomly vary a value x != 0 by
    y = e^log(x +/- log_wiggle) +/- absolute_wiggle
    keeping the sign
    """
    log_wiggled = sign(x) * exp(rng.normal(log(abs(x)), log_wiggle))
    return rng.normal(log_wiggled, absolute_wiggle)
