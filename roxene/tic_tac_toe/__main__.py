from random import Random
from sys import argv

import tensorflow as tf

from .runner import Runner

SEED = 11235

import logging

logging.basicConfig(level=logging.DEBUG)

rigged_rng = Random(SEED)
tf.random.set_seed(SEED)

num_organisms = int(argv[1])
num_iterations = int(argv[2])
runner = Runner(num_organisms, rigged_rng)

# Start trials and do GA stuff in a single-threaded alternating loop
for iteration in range(num_iterations):
    print("Running a trial")
    trial = runner.run_trial()
    print(f"Game finished with moves {[(move.letter, move.position, move.outcomes) for move in trial.moves]}")
