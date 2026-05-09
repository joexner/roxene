import argparse
import logging
import sys
import time
from sqlalchemy import create_engine, text
from threading import Thread

from numpy.random import Generator, default_rng
from sqlalchemy import create_engine

from .environment import Environment
from ..persistence import EntityBase
from ..util import set_rng

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(threadName)s]\t- %(name)s: %(message)s',
                    force=True)
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

# Create a fresh Postgres database for this run and initialize schema
admin_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
db_name = f"roxene_{int(time.time())}"
logger.info(f"Creating database {db_name}")
admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
with admin_engine.connect() as conn:
    conn.execute(text(f'CREATE DATABASE "{db_name}"'))
admin_engine.dispose()

db_url = f"postgresql+psycopg2://postgres:postgres@localhost:5432/{db_name}"
engine = create_engine(db_url)
EntityBase.metadata.create_all(engine)

logger.info(f"Seed={SEED}")
main_rng: Generator = default_rng(SEED)
set_rng(main_rng)
env = Environment(engine)

logger.info(f"Populating environment with {num_organisms} organisms and {num_mutagens} mutagens")
env.populate(num_organisms)
env.add_mutagens(num_mutagens)
logger.info("Done populating environment")

# Replace 5% of the herd at a time, up to 5
num_to_cull = num_to_breed = int(max(num_organisms * .05, 5))

def run(worker_trials: int, worker_rng: Generator, worker_logger: logging.Logger):
    set_rng(worker_rng)
    for iteration in range(worker_trials):
        worker_logger.info(f"Building trial {iteration}")
        trial = env.start_trial()
        worker_logger.info(f"Starting trial {iteration} between players {trial.participants[0]} and {trial.participants[1]}")
        trial.run()
        worker_logger.info(f"Trial {iteration} complete, saving results")
        env.complete_trial(trial)
        worker_logger.info(f"Finished trial {iteration} with moves {[(move.letter, move.position, move.outcomes) for move in trial.moves]}")
        if iteration % args.breed_and_cull_interval == 0 and iteration > 0:
            worker_logger.info("Culling")
            env.cull(num_to_cull)
            worker_logger.info("Done culling, breeding")
            env.breed(num_to_breed)
            worker_logger.info("Done breeding")

num_threads = 10
threads = []
rngs = main_rng.spawn(num_threads)

for i in range(num_threads):
    logger.info(f"Starting thread {i}")
    thread = Thread(
        target=run,
        args=(
            int(( num_trials - 1 ) / num_threads ) + 1, # Estimate, total could be over by (num_threads - 1)
            rngs.pop(),
            logger.getChild('_' + str(i)),
        )
    )
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()



# mlflow.log_metric("trials_executed", num_trials)

# with open('runner.pickle', 'wb') as file:
#     pickle.dump(env, file)

# mlflow.log_artifact('runner.pkl')
