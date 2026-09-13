"""Block until the init Job has populated the organism pool.

Used as a pre-flight gate by the worker and breeder. Both need a non-empty
pool before they can do anything useful, and there is no k8s primitive that
makes a StatefulSet/Deployment wait for a Job -- so we poll.

Deliberately stdlib-only (plus urllib for the API call): this runs as the
entrypoint of the roxene image, which is python:3.12-slim. The obvious
alternative is an initContainer running `kubectl`, but that needs an image
with a shell, kubectl AND psql, which rules out the distroless official
registry.k8s.io/kubectl and adds a moving part that can fail to pull.
Reading the mounted ServiceAccount token is one fewer dependency.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# Mounted into every pod automatically unless automountServiceAccountToken: false
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def _api_get(path: str) -> dict | None:
    """GET a Kubernetes API path using the pod's ServiceAccount token.

    Returns None if the call fails, so the caller can just retry rather
    than having to distinguish 404 (no Job yet) from a transient 500.
    """
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT")
    if not host or not port:
        logger.info("Not running in Kubernetes (no KUBERNETES_SERVICE_HOST), skipping gate")
        return None
    try:
        with open(TOKEN_PATH) as f:
            token = f.read().strip()
    except OSError:
        logger.info("No ServiceAccount token mounted, skipping gate")
        return None

    url = f"https://{host}:{port}{path}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, cafile=CA_PATH, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # The Role in worker-rbac.yaml is missing or mis-scoped. Surface it
            # loudly: silently retrying would park the pod in Init forever.
            logger.error(f"Forbidden reading {path} -- check the wait-for-init Role/RoleBinding")
        logger.debug(f"API GET {path} returned {e.code}")
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        logger.debug(f"API GET {path} failed: {e}")
        return None


def job_finished(job_name: str, namespace: str) -> bool:
    """True once the Job has a Complete or Failed condition."""
    job = _api_get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
    if job is None:
        return False
    for condition in job.get("status", {}).get("conditions", []):
        if condition.get("type") in ("Complete", "Failed") and condition.get("status") == "True":
            logger.info(f"Job {job_name} is {condition['type']}")
            return True
    return False


def pool_populated(db_url: str | None) -> bool:
    """True if at least one living organism exists."""
    if not db_url:
        return False
    try:
        engine = create_engine(db_url, pool_size=1)
        try:
            with engine.connect() as conn:
                count = conn.execute(
                    text("SELECT count(*) FROM organism WHERE deleted_date IS NULL")
                ).scalar()
            logger.info(f"Organism pool has {count} organisms")
            return bool(count)
        finally:
            engine.dispose()
    except Exception as e:
        # Usually "relation does not exist" -- the init Job hasn't created the
        # schema yet. Retry rather than crash.
        logger.debug(f"Pool check failed: {e}")
        return False


def db_ready(db_url: str | None) -> bool:
    """True if the database accepts a trivial query.

    Postgres needs time for initdb on first boot, and init connects through
    pgbouncer, which is a second hop. Nothing in k8s gates a Job on another
    workload's readinessProbe, so `kubectl apply -k` starts the init Job at
    the same instant as postgres and init dies with OperationalError if it
    gets there first. Previously that just crashed (exit 1) and relied on
    backoffLimit to retry.
    """
    if not db_url:
        return False
    try:
        engine = create_engine(db_url, pool_size=1)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        finally:
            engine.dispose()
    except Exception as e:
        logger.debug(f"Database not ready: {e}")
        return False


def wait_for_db(db_url: str | None, interval: float = 2.0, timeout: float = 300.0) -> None:
    """Block until the database accepts a trivial query.

    Raises TimeoutError if timeout elapses, so a genuinely broken DATABASE_URL
    surfaces as a real error instead of an infinite hang.
    """
    deadline = time.monotonic() + timeout
    logger.info("Waiting for the database to accept connections")
    while True:
        if db_ready(db_url):
            logger.info("Database is ready")
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out after {timeout}s waiting for the database")
        time.sleep(interval)


def wait_for_init(db_url: str | None, job_name: str = "init", namespace: str | None = None,
                  interval: float = 5.0, timeout: float | None = None) -> None:
    """Poll until the init Job is done and the pool is non-empty.

    The Job condition alone isn't enough: it flips to Complete when the
    container exits, and the organism rows are committed inside that same
    process, so in practice checking the pool is what actually matters. The
    Job check is kept because it distinguishes "init failed" from "init
    still running" in the logs.

    Raises TimeoutError if timeout elapses.
    """
    if namespace is None:
        namespace = os.environ.get("ROXENE_NAMESPACE", "roxene")

    deadline = None if timeout is None else time.monotonic() + timeout
    logger.info(f"Waiting for init Job {namespace}/{job_name} and a non-empty organism pool")
    while True:
        if job_finished(job_name, namespace) and pool_populated(db_url):
            logger.info("Init gate satisfied")
            return
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for init Job {job_name} / organism pool"
            )
        time.sleep(interval)
