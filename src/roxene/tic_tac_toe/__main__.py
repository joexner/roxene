import argparse
import logging
import sys
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from . import Trial
from .environment import Environment
from .. import EntityBase

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Play some tic-tac-toe')

parser.add_argument('pool_size', type=int, help='Number of cloned Organisms initially in the pool')
parser.add_argument('num_trials', type=int, help='Number of tic-tac-toe trials to run')
parser.add_argument('--breed_and_cull_interval', type=int, help='Number of trials between rounds of culling and breeding', default=10)
parser.add_argument('--num_mutagens', type=int, help='Number of mutagens in the pool', default=100)

args = parser.parse_args(sys.argv[1:])

num_organisms = args.pool_size
num_mutagens = args.num_mutagens
num_trials = args.num_trials
SEED = 11235

# import mlflow
# import pickle
# mlflow.log_params({
#     'num_organisms': num_organisms,
#     'num_mutagens': num_mutagens,
#     'num_trials': num_trials,
#     'seed': SEED
# })

filename = "sqlite:///run_%d.db" % int(time.time())
engine = create_engine(filename)
EntityBase.metadata.create_all(engine)


env = Environment(
    seed=SEED,
    engine=engine,
)

logger.info(f"Populating environment with {num_organisms} organisms and {num_mutagens} mutagens")
env.populate(num_organisms)
env.add_mutagens(num_mutagens)
logger.info("Done populating environment")

# Replace 5% of the herd at a time, up to 5
num_to_cull = num_to_breed = int(max(num_organisms * .05, 5))

# Start trials and do GA stuff in a single-threaded alternating loop
for iteration in range(1, num_trials + 1):
    trial: Trial = env.start_trial()
    logger.info(f"Starting trial {iteration} between players {trial.participants[0]} and {trial.participants[1]}")
    trial.run()
    env.complete_trial(trial)
    logger.info(f"Finished trial {iteration} with moves {[(move.letter, move.position, move.outcomes) for move in trial.moves]}")
    if iteration % args.breed_and_cull_interval == 0:
        logger.info("Breeding and culling")
        env.cull(num_to_cull)
        env.breed(num_to_breed)

# mlflow.log_metric("trials_executed", num_trials)

# with open('runner.pickle', 'wb') as file:
#     pickle.dump(env, file)

# mlflow.log_artifact('runner.pkl')
