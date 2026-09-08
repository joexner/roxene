import argparse
import logging
import os
import re
import signal
import time
from threading import Event, Thread

from numpy.random import Generator, default_rng
from sqlalchemy import create_engine, text

from .environment import Environment
from ..persistence import EntityBase
from ..util import set_rng

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Play some tic-tac-toe')

parser.add_argument('--role', choices=['init', 'worker', 'breeder', 'reaper'], default='worker',
                    help='Which job this process does: init populates the pool, worker runs trials, breeder culls and breeds, reaper deletes trials abandoned by dead workers')
# Flags, not positionals: pool_size and num_trials are used by different roles
# (init vs worker) and are never both passed, so two `nargs='?'` positionals
# would make a single trailing positional ambiguous.
parser.add_argument('--pool_size', type=int, default=1000,
                    help='Number of organisms in the pool (init)')
parser.add_argument('--num_trials', default='forever',
                    help='Number of trials to run (worker), or "forever"')
parser.add_argument('--num_threads', type=int, default=1,
                    help='Number of threads to use to run trials')
parser.add_argument('--breed_and_cull_interval', type=int, default=10,
                    help='Seconds between rounds of culling and breeding (breeder)')
parser.add_argument('--num_mutagens', type=int, default=100,
                    help='Number of mutagens in the pool (init)')
parser.add_argument('--stale_after_minutes', type=int, default=15,
                    help='Trials running longer than this are considered abandoned (reaper)')
parser.add_argument("--db_url",
                    help='Database URL', default=os.environ.get('DATABASE_URL'))


def parse_num_trials(value: str) -> int | None:
    """Return the number of trials to run, or None for 'forever'."""
    return None if value == 'forever' else int(value)


def make_engine(db_url: str | None, num_threads: int):
    if not db_url:
        # Create a fresh Postgres database for this run and initialize schema
        admin_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
        db_name = f"roxene_{int(time.time())}"
        logger.info(f"Creating database {db_name}")
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        admin_engine.dispose()
        db_url = f"postgresql+psycopg2://postgres:postgres@localhost:5432/{db_name}"
    return create_engine(db_url, pool_size=max(num_threads, 2))


def run_init(args, env, seed: int) -> None:
    """Populate the organism pool and add mutagens (idempotent)."""
    num_organisms = args.pool_size
    if env.count_organisms() > 0:
        logger.info(f"Pool already has {env.count_organisms()} organisms, skipping population")
    else:
        logger.info(f"Populating environment with {num_organisms} organisms and {args.num_mutagens} mutagens")
        set_rng(default_rng(seed))
        env.populate(num_organisms)
        env.add_mutagens(args.num_mutagens)
        logger.info("Done populating environment")


def run_reaper(args, engine) -> None:
    """Delete trials abandoned by dead workers so their organisms return to the pool."""
    # Trials whose worker died (e.g. spot preemption) never get an end_date.
    # Their moves were never persisted, so deleting the rows is safe, and it
    # returns the organisms to the idle pool automatically.
    with engine.begin() as conn:
        stale_trials = (
            "SELECT id FROM trial"
            "  WHERE end_date IS NULL AND start_date < now() - make_interval(mins => :stale)"
        )
        conn.execute(text(f"DELETE FROM move WHERE trial_id IN ({stale_trials})"), {"stale": args.stale_after_minutes})
        conn.execute(text(f"DELETE FROM player WHERE trial_id IN ({stale_trials})"), {"stale": args.stale_after_minutes})
        result = conn.execute(text(f"DELETE FROM trial WHERE end_date IS NULL AND start_date < now() - make_interval(mins => :stale) RETURNING id"), {"stale": args.stale_after_minutes})
        reaped = [row[0] for row in result]
    logger.info(f"Reaped {len(reaped)} abandoned trials: {reaped}")


def run_breeder(args, env, seed: int) -> None:
    """Repeatedly cull the weakest organisms and breed their replacements."""
    num_organisms = env.count_organisms()
    # Replace 5% of the herd at a time, up to 5
    num_to_cull = num_to_breed = int(max(num_organisms * 0.05, 5))
    set_rng(default_rng(seed))
    logger.info(f"Breeder started: culling/breeding {num_to_cull}/{num_to_breed} every {args.breed_and_cull_interval}s")
    try:
        while True:
            time.sleep(args.breed_and_cull_interval)
            logger.info("Culling")
            env.cull(num_to_cull)
            logger.info("Done culling, breeding")
            env.breed(num_to_breed)
            logger.info("Done breeding")
    except KeyboardInterrupt:
        pass


def run_worker(env, num_trials: int | None, num_threads: int, seed: int) -> None:
    """Run trials across worker threads until stopped (by count or signal)."""
    stop_event = Event()

    def handle_sigterm(signum, _frame):
        # Spot preemption / rollout: finish the current trial, then exit.
        logger.info(f"Got signal {signum}, will stop after the current trial")
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    def run_trials(worker_trials: int | None, worker_rng: Generator) -> None:
        set_rng(worker_rng)
        iteration = 0
        while not stop_event.is_set() and (worker_trials is None or iteration < worker_trials):
            logger.info("Building trial")
            trial = env.start_trial()
            logger.info(f"Starting trial, {env.count_trials(True, False)} trials running, {env.count_trials()} trials total")
            trial.run()
            logger.info("Trial complete, saving results")
            env.complete_trial(trial)
            logger.info(f"Finished trial {trial} with {len(trial.moves)} moves")
            iteration += 1

    # Distribute the trials across threads: ceil(num_trials / num_threads) each,
    # so the total could overshoot by up to (num_threads - 1).
    per_thread = None if num_trials is None else int((num_trials - 1) / num_threads) + 1
    main_rng: Generator = default_rng(seed)
    rngs = main_rng.spawn(num_threads)

    threads = []
    for i in range(num_threads):
        logger.info(f"Starting thread {i}")
        thread = Thread(target=run_trials, args=(per_thread, rngs[i]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # with open('runner.pickle', 'wb') as file:
    #     pickle.dump(env, file)


def main() -> None:
    # Configured here rather than at import time so that importing this module
    # (e.g. from a notebook) doesn't hijack logging. This has to happen for the
    # CLI/container entrypoint though: with no handler installed, Python falls
    # back to logging.lastResort, which drops everything below WARNING -- which
    # is why the k8s pods emitted no output at all.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - [%(threadName)s]\t- %(name)s: %(message)s",
        force=True,
    )
    args = parser.parse_args()
    num_trials = parse_num_trials(args.num_trials)

    # Per-process seed: base seed + this pod's ordinal, so every process in a
    # StatefulSet gets a distinct (and reproducible) UUID stream. In a k8s pod the
    # ordinal is the suffix of the hostname (e.g. roxene-worker-7 -> 7).
    seed_offset_env = os.environ.get('SEED_OFFSET')
    if seed_offset_env is not None:
        seed_offset = int(seed_offset_env)
    else:
        m = re.search(r"-(\d+)$", os.environ.get("HOSTNAME", ""))
        seed_offset = int(m.group(1)) if m else 0
    seed = 11235 + seed_offset
    logger.info(f"Role={args.role}, seed offset={seed_offset}")

    engine = make_engine(args.db_url, args.num_threads)
    try:
        EntityBase.metadata.create_all(engine)
        env = Environment(engine)
        match args.role:
            case 'init':
                run_init(args, env, seed)
            case 'reaper':
                run_reaper(args, engine)
            case 'breeder':
                run_breeder(args, env, seed)
            case 'worker':
                run_worker(env, num_trials, args.num_threads, seed)
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
