"""Early, configurable workspace plans; no allocation or device probing.

Plans account for named live buffers, not process RSS. The caller's table,
Python objects and private BLAS/allocator workspace are not bounded by this
contract. Estimates must be supplied before constructing their buffers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
import os
import re

from openecon.analysis_contracts import AnalysisError

DEFAULT_WORKSPACE_MB = 512
_override: ContextVar[int | None] = ContextVar("openecon_workspace_bytes", default=None)
_SCOPE = "estimated live model-input and tensor buffers; excludes caller input, Python result objects, private BLAS workspace and allocator overhead; not a process-RSS limit"


def _count(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < int(positive):
        raise AnalysisError("invalid_resource_budget", f"{name} must be a {'positive' if positive else 'nonnegative'} integer.")
    return int(value)


def tensor_bytes(shape: Sequence[int], *, itemsize: int = 8) -> int:
    """Exact Python-integer arithmetic, including dimensions above int64."""
    size = _count(itemsize, "itemsize", positive=True)
    for dimension in shape:
        size *= _count(dimension, "tensor dimension")
    return size


def workspace_budget_bytes() -> int:
    """Current task-local override, or OPENECON_WORKSPACE_MB (default 512)."""
    override = _override.get()
    if override is not None:
        return override
    raw = os.environ.get("OPENECON_WORKSPACE_MB", str(DEFAULT_WORKSPACE_MB))
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) < 1:
        raise AnalysisError("invalid_resource_budget", "OPENECON_WORKSPACE_MB must be a positive integer number of MiB.")
    return int(raw) * 1024**2


@contextmanager
def use_workspace_budget(memory_mb: int):
    """Temporarily set this thread/task's workspace budget, restoring on exit."""
    token = _override.set(_count(memory_mb, "memory_mb", positive=True) * 1024**2)
    try:
        yield
    finally:
        _override.reset(token)


@dataclass(frozen=True)
class ResourcePlan:
    operation: str
    budget_bytes: int
    buffers: tuple[tuple[str, int], ...]

    @property
    def estimated_bytes(self) -> int:
        return sum(size for _, size in self.buffers)

    def record(self) -> dict:
        return {"operation": self.operation, "estimated_workspace_bytes": self.estimated_bytes,
                "budget_bytes": self.budget_bytes, "buffers": dict(self.buffers), "scope": _SCOPE}


def plan_workspace(operation: str, buffers: Mapping[str, int], *, budget_bytes: int | None = None) -> ResourcePlan:
    """Check a conservative live-buffer plan before allocating any named block."""
    if not isinstance(operation, str) or not operation or not isinstance(buffers, Mapping):
        raise AnalysisError("invalid_resource_budget", "Supply an operation name and named buffer estimates.")
    budget = workspace_budget_bytes() if budget_bytes is None else _count(budget_bytes, "budget_bytes", positive=True)
    entries = []
    for name, size in buffers.items():
        if not isinstance(name, str) or not name:
            raise AnalysisError("invalid_resource_budget", "Workspace buffers need nonempty names.")
        entries.append((name, _count(size, f"buffer {name}")))
    plan = ResourcePlan(operation, budget, tuple(entries))
    if plan.estimated_bytes > budget:
        largest = sorted(entries, key=lambda item: item[1], reverse=True)[:3]
        detail = ", ".join(f"{name}={size:,} bytes" for name, size in largest)
        error = AnalysisError("workspace_limit", f"{operation} needs an estimated {plan.estimated_bytes:,} workspace bytes, exceeding the {budget:,}-byte budget ({detail}). Reduce model dimensions or set OPENECON_WORKSPACE_MB to a suitable budget. This is a buffer estimate, not total process memory.")
        error.resource_plan = plan.record()
        raise error
    return plan
