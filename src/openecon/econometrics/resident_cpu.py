"""Pin resident numerical APIs to their declared CPU domain locally."""

from functools import wraps
import torch


def resident_cpu(function):
    """Do not inherit a caller's unrelated default tensor device."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.device("cpu"):
            return function(*args, **kwargs)

    return wrapped
