# syntax=docker/dockerfile:1
# Stage "training": roxene + torch, used by workers/breeder/init/reaper.
FROM python:3.12-slim AS training

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Default to a single worker process; the k8s manifests pass --role explicitly
ENTRYPOINT ["python", "-m", "roxene.tic_tac_toe"]

# Stage "notebook": training image + the jupyter stack (~227MB).
# Built as the default target; the k8s notebook deployment overrides the
# command, so the inherited entrypoint is never used here.
FROM training AS notebook

RUN pip install jupyter

WORKDIR /app
COPY notebooks ./notebooks
