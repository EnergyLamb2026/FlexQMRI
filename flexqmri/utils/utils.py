'''Basic utils functions.'''

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def make_serializable(obj):
    """Make config JSON-serializable.

    Args:
        obj: Object to convert (dict, list, ndarray, scalar, etc.)

    Returns:
        JSON-serializable version of *obj*.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    else:
        return str(obj)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed (int): Seed value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
