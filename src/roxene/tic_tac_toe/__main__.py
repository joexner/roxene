import argparse
import logging
import os
import re
import signal
import time
from datetime import datetime, timedelta
from threading import Event, Thread

from numpy.random import Generator, default_rng
from sqlalchemy import create_engine, text, update, Engine
from sqlalchemy.orm import Session

from . import metrics
from .environment import BreederState, Environment
from .init_gate import wait_for_db, wait_for_init
from .trial import Trial
from ..persistence import EntityBase
from ..util import set_rng

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Play some tic-tac-toe')

parser.add_argument('--role', choices=['init', 'worker', 'breeder', 'reaper'], default='worker',
                    help='Which job this process does: init populates the pool, worker runs trials, breeder culls and breeds, reaper finishes off trials abandoned by dead workers')
parser.add_argument('--pool_size', type=int, default=1000,              help='Number of organisms in the pool (init)')
parser.add_argument('--num_mutagens', type=int, default=100,            help='Number of mutagens to put in the pool (init only)')
parser.add_argument('--num_trials', default='forever',                  help='Number of trials to run (worker), or "forever"')
parser.add_argument('--num_threads', type=int, default=5,               help='Number of threads to use to run trials.')
parser.add_argument('--breed_every_n_trials', type=int, default=50,      help='Breeder: run a cull+breed cycle once this many trials have completed since the last one, so the run frequency scales with the trial completion rate (breeder)')
parser.add_argument('--stale_after_minutes', type=int, default=60,      help='Trials running longer than this are considered abandoned (reaper only)')
parser.add_argument("--db_url", default=os.environ.get('DATABASE_URL'), help='Database URL')
parser.add_argument("--metrics_port", type=int, default=None,           help='Port to serve /metrics on (worker/breeder). Defaults to the METRICS_PORT env var; 0 disables.')
parser.add_argument("--wait_for_init", action="store_true",             help='Block until the init Job has populated the organism pool before starting (worker/breeder)')
parser.add_argument("--init_timeout", type=float, default=None,         help='Seconds to wait for --wait_for_init before giving up. Default: wait forever')
parser.add_argument("--db_timeout", type=float, default=300.0,          help='Seconds to wait for the database to accept connections before giving up')



def make_engine(db_url: str | None, num_threads: int) -> Engine:
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
    return create_engine(db_url, pool_size=num_threads)


def run_init(env: Environment, num_organisms: int, num_mutagens: int, seed: int | None) -> None:
    """Populate the organism pool and add mutagens (idempotent)."""
    if env.count_organisms() > 0:
        logger.info(f"Pool already has {env.count_organisms()} organisms, skipping population")
    else:
        logger.info(f"Populating environment with {num_organisms} organisms and {num_mutagens} mutagens")
        set_rng(default_rng(seed))
        env.populate(num_organisms)
        env.add_mutagens(num_mutagens)
        logger.info("Done populating environment")


def run_reaper(args, engine) -> None:
    """Mark trials abandoned by dead workers as finished so their organisms return to the pool."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=args.stale_after_minutes)
    with (Session(engine) as session):
        reaped = session.scalars(
            update(Trial)
            .where(Trial.end_date.is_(None), Trial.start_date < cutoff)
            .values(end_date=now)
            .returning(Trial.id)
        ).all()
        session.commit()
    logger.info(f"Marked {len(reaped)} abandoned trials as finished: {reaped}")


def run_breeder(args, env, seed: int, metrics_port: int) -> None:
    metrics.start_server(metrics_port, env)
    set_rng(default_rng(seed))
    logger.info("Breeder starting")

    completed_total = env.count_trials(completed=True)
    with env.sessionmaker() as session:
        state = session.get(BreederState, 1)
        if state is None:
            # First run after init: base the counter on the current total so we
            # breed once --breed_every_n_trials NEW trials have finished.
            state = BreederState(id=1, trials_completed_at_last_breed=completed_total)
            session.add(state)
            session.commit()
        since_last_breed = max(0, completed_total - state.trials_completed_at_last_breed)
        last_breed_at = state.last_breed_at

    if since_last_breed < args.breed_every_n_trials:
        logger.info(
            f"Skipping cull+breed cycle: {since_last_breed}/{args.breed_every_n_trials} "
            f"trials completed since last cycle (last ran {last_breed_at})"
        )
        return

    logger.info(f"Running cull+breed cycle ({since_last_breed} trials completed since last cycle)")
    culled = env.cull()
    if culled:
        metrics.record_cull()
    bred = env.breed()
    if bred:
        metrics.record_breed()
    logger.info(f"Done (culled={culled}, bred={bred})")

    # Record the high-water mark after the cycle so a crash mid-cycle re-runs
    # the cycle instead of silently skipping the next --breed_every_n_trials.
    with env.sessionmaker() as session:
        state = session.get(BreederState, 1)
        state.trials_completed_at_last_breed = completed_total
        state.last_breed_at = datetime.now()
        session.commit()


def run_worker(env, num_trials: int | None, num_threads: int, seed: int | None, metrics_port: int) -> None:
    """Run trials across worker threads until stopped (by count or signal)."""

    # Start the metrics server but don't keep updating trial counts and pool size metrics
    metrics.start_server(metrics_port, None)

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
            t0 = time.monotonic()
            trial.run()
            duration = time.monotonic() - t0
            logger.info("Trial complete, saving results")
            env.complete_trial(trial)
            metrics.record_trial(trial, duration)
            logger.info(f"Finished trial {trial} with {len(trial.moves)} moves in {duration:.3f}s")
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

    # Set up logging
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(levelname)s - [%(threadName)s]\t- %(name)s: %(message)s",
        force=True,
    )

    # Parse args
    args = parser.parse_args()

    # 0 disables the metrics server
    metrics_port = args.metrics_port
    if metrics_port is None:
        metrics_port = int(os.environ.get("METRICS_PORT") or 0)

    # Trial limit before the workers stop, default = unlimited
    num_trials = int(args.num_trials) if args.num_trials and args.num_trials != 'forever' else None

    # Per-process seed: base seed + this pod's ordinal, so every process in a
    # StatefulSet gets a distinct (and reproducible) UUID stream. In a k8s pod the
    # ordinal is the suffix of the hostname (e.g. roxene-worker-7 -> 7).
    m = re.search(r"-(\d+)$", os.environ.get("HOSTNAME", ""))
    seed_offset = int(m.group(1)) if m else 0
    seed = 11235 + seed_offset
    logger.info(f"Role={args.role}, seed offset={seed_offset}")

    # Wait for Postgres and pgbouncer
    wait_for_db(args.db_url, timeout=args.db_timeout)

    # Wait for the init job to finish populating the environment
    if args.wait_for_init:
        wait_for_init(args.db_url, timeout=args.init_timeout)

    engine: Engine = make_engine(args.db_url, args.num_threads)
    try:
        env = Environment(engine)
        match args.role:
            case 'init':
                EntityBase.metadata.create_all(engine)
                run_init(env, args.pool_size, args.num_mutagens, seed)
            case 'worker':
                run_worker(env, num_trials, args.num_threads, seed, metrics_port)
            case 'reaper':
                run_reaper(args, engine)
            case 'breeder':
                # The breeder is a CronJob and needs its control table to exist
                # even on a database created before the table was added.
                EntityBase.metadata.create_all(engine)
                run_breeder(args, env, seed, metrics_port)
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
