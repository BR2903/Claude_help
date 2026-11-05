"""One place that fixes every source of randomness.

Randomness in this project comes from four places, and all are seeded:
  1. Python's `random`          -> random.seed(seed)
  2. NumPy's global RNG         -> np.random.seed(seed)   (agents also own a np.random.Generator)
  3. PyTorch (weight init, ...) -> torch.manual_seed(seed)
  4. The environment            -> env.reset(seed=seed) on the FIRST reset and env.action_space.seed(seed)
     (later resets without a seed continue the env's own seeded RNG stream, which is deterministic).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # tabular-only runs do not need torch
        pass
