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
