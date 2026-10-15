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


# Stage "training-rocm": roxene on the official ROCm base image, for the AMD GPU
# worker on enzo.local. The RX 7600 XT is gfx1102, which ROCm 10.0 targets
# natively (no HSA_OVERRIDE_GFX_VERSION workaround needed, it's just pinned).
# Separate target so the CPU image used by GKE stays small.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0 AS training-rocm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.2 \
    ROCR_VISIBLE_DEVICES=0 \
    HIP_VISIBLE_DEVICES=0

COPY pyproject.toml ./
COPY src ./src

# Ubuntu marks the system python externally managed (PEP 668); install into a
# venv that still sees the image's ROCm build of torch.
RUN python3 -m venv /opt/venv --system-site-packages && /opt/venv/bin/pip install .
ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["python", "-m", "roxene.tic_tac_toe"]
