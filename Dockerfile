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

# CPU-only cp314t torch keeps the image small (PyPI's linux wheel ships CUDA).
# Pinned to 2.13.0 to match the ROCm stage and the VM benchmark env.
RUN uv pip install --python /opt/venv/bin/python "torch==2.13.0+cpu" \
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


# Stage "training-rocm": roxene on the official ROCm base image, for the AMD GPU
# worker on enzo.local. The RX 7600 XT is gfx1102, which ROCm 10.0 targets
# natively (no HSA_OVERRIDE_GFX_VERSION workaround needed, it's just pinned).
# Separate target so the CPU image used by GKE stays small.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0 AS training-rocm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HSA_OVERRIDE_GFX_VERSION=11.0.2 \
    ROCR_VISIBLE_DEVICES=0 \
    HIP_VISIBLE_DEVICES=0

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# No rocm/pytorch image ships a free-threaded python and there is no standalone
# cp314t wheel for ROCm 10.0, so layer the self-contained torch 2.13.0+rocm7.2
# cp314t wheel (it bundles its own userspace ROCm; the host only needs the
# amdgpu driver + /dev/kfd) into a 3.14t venv.
RUN uv python install 3.14t \
    && uv venv --python 3.14t /opt/venv-ft \
    && uv pip install --python /opt/venv-ft/bin/python "torch==2.13.0+rocm7.2" \
        --extra-index-url https://download.pytorch.org/whl/rocm7.2

COPY pyproject.toml ./
COPY src ./src
RUN uv pip install --python /opt/venv-ft/bin/python .
ENV PATH="/opt/venv-ft/bin:$PATH"

ENTRYPOINT ["python", "-m", "roxene.tic_tac_toe"]
