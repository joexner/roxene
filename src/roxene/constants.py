import os

import numpy
import torch

TORCH_PRECISION = torch.float16
NP_PRECISION = numpy.float16

# Where organism state lives. On a ROCm container this picks the HIP device
# (torch exposes it as "cuda"); with no GPU it stays on the CPU, so GKE and the
# unit tests behave exactly as before. Override with ROXENE_DEVICE=cpu if you
# want to force the CPU on a GPU node.
DEVICE = torch.device(os.environ.get("ROXENE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))

# Cap torch's intra-op threads. Many worker threads each spawning 8 OpenMP
# threads oversubscribes the box (10x8=80 on 16 CPUs); 2 is the measured
# sweet spot. Override with ROXENE_TORCH_THREADS.
torch.set_num_threads(int(os.environ.get("ROXENE_TORCH_THREADS", "2")))

# Optional numpy fp32 fast path for small organisms (isolated benches: ~4x
# faster than torch CPU, ~13x faster than torch GPU at small matrix sizes).
# "off" keeps the torch path everywhere (tests); "on" uses numpy for organisms
# under the flop threshold; "auto" = "on".
NUMPY_BACKEND = os.environ.get("ROXENE_NUMPY_BACKEND", "off").lower()
# numpy is used when dim_C * total_in_fb <= this (0 disables it)
NUMPY_BACKEND_MAX_FLOPS = int(os.environ.get("ROXENE_NUMPY_BACKEND_MAX_FLOPS", "200000"))
