# syntax=docker/dockerfile:1
# Stage "training": roxene + torch on free-threaded CPython 3.14t (GIL off),
# used by workers/breeder/init/reaper and as the base for the notebook.
FROM debian:bookworm-slim AS training

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# uv manages the interpreter; 3.14t is the free-threaded build (GIL disabled).
RUN uv python install 3.14t && uv venv --python 3.14t /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch keeps the image small (PyPI's linux wheel ships CUDA).
# Pinned to the version in uv.lock.
RUN uv pip install --python /opt/venv/bin/python "torch==2.15.0+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python .

# Default to a single worker process; the k8s manifests pass --role explicitly
ENTRYPOINT ["python", "-m", "roxene.tic_tac_toe"]

# Stage "notebook": training image + the jupyter stack (~227MB).
# Built as the default target; the k8s notebook deployment overrides the
# command, so the inherited entrypoint is never used here.
FROM training AS notebook

RUN uv pip install --python /opt/venv/bin/python jupyter

WORKDIR /app
COPY notebooks ./notebooks
