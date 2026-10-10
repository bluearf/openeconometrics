"""Scoped test cleanup for Torch's actual default-device override.

An implicit CPU default has no DeviceContext. Restoring the value returned by
get_default_device() would install one and add dispatch overhead to later tests.
The locked test runtimes use Torch 2.14; private context access here preserves
the original object and surrounding mode identities, including explicit CPU.
This helper is intentionally test-only and is never an autouse fixture.
"""

from contextlib import contextmanager

import torch
from torch.overrides import _get_current_function_mode_stack


@contextmanager
def preserve_torch_default_device():
    """Restore the exact prior override after a scoped set_default_device test."""
    global_context = torch._GLOBAL_DEVICE_CONTEXT
    had_attribute = hasattr(global_context, "device_context")
    original = getattr(global_context, "device_context", None)
    original_modes = tuple(_get_current_function_mode_stack())
    try:
        yield
    finally:
        torch.set_default_device(None)
        if original is not None:
            original.__enter__()
        if had_attribute:
            global_context.device_context = original
        else:
            del global_context.device_context
        restored_modes = tuple(_get_current_function_mode_stack())
        assert len(restored_modes) == len(original_modes)
        assert all(actual is expected for actual, expected in zip(restored_modes, original_modes))
