import os
import string
import time

import numpy as np
import torch


def tempfile(suffix=None, fmt="bin", loc="/tmp"):
    """Generate name for a temporary file."""
    letters = "".join(np.random.choice(list(string.ascii_letters), size=6))
    numbers = time.time_ns() % 1000_000
    if suffix is None:
        path = os.path.join(loc, f"{letters:s}-{numbers:06d}.{fmt}")
    else:
        path = os.path.join(loc, f"{letters:s}-{numbers:06d}-{suffix}.{fmt}")

    return tempfile(suffix, fmt) if os.path.exists(path) else path


def corr(a: np.ndarray, b: np.ndarray, axis=None):
    """Compute Pearson's correlation along specified axis."""
    a_mean = a.mean(axis=axis, keepdims=True)
    b_mean = b.mean(axis=axis, keepdims=True)
    a, b = (a - a_mean), (b - b_mean)

    a_sum2 = (a**2).sum(axis=axis, keepdims=True)
    b_sum2 = (b**2).sum(axis=axis, keepdims=True)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a, b = (a / np.sqrt(a_sum2)), (b / np.sqrt(b_sum2))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = (a / torch.sqrt(a_sum2)), (b / torch.sqrt(b_sum2))
    else:
        raise TypeError(f"Incompatible types: {type(a)} and {type(b)}")

    return (a * b).sum(axis=axis)
