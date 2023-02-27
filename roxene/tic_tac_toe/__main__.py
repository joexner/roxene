import argparse
import logging
import sys

import mlflow
import pickle

from .runner import Runner

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Play some tic-tac-toe')

parser.add_argument('pool_size', type=int, help='Number of cloned Organisms initially in the pool')
parser.add_argument('num_trials', type=int, help='Number of tic-tac-toe trials to run')
parser.add_argument('--breed_and_cull_interval', type=int,
                    help='Number of trials between rounds of culling and breeding', default=10)
parser.add_argument('--num_mutagens', type=int, help='Number of mutagens in the pool', default=100)

args = parser.parse_args(sys.argv[1:])

num_organisms = args.pool_size
num_mutagens = args.num_mutagens
num_trials = args.num_trials
SEED = 11235

mlflow.log_params({
    'num_organisms': num_organisms,
    'num_mutagens': num_mutagens,
    'num_trials': num_trials,
    'seed': SEED
})

runner = Runner(num_organisms=num_organisms, num_mutagens=num_mutagens, seed=SEED)

num_to_cull = num_to_breed = int(max(num_organisms * .05, 5))  # Replace 5% of the herd at a time, up to 5

# Start trials and do GA stuff in a single-threaded alternating loop
for iteration in range(num_trials):
    trial = runner.run_trial()
    logger.info(f"Game finished with moves {[(move.letter, move.position, move.outcomes) for move in trial.moves]}")
    if iteration % args.breed_and_cull_interval == 0:
        logger.info("Breeding and culling")
        runner.cull(num_to_cull)
        runner.breed(num_to_breed)
        runner.completed_trials.append(trial)

mlflow.log_metric("trials_executed", num_trials)

with open('runner.pickle', 'wb') as file:
    pickle.dump(runner, file)

mlflow.log_artifact('runner.pkl')
