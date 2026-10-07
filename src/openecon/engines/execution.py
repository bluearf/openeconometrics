"""Bounded acceleration with float64 inference and checked Metal refinement.

CUDA factors use float64. Metal float32 QR supplies a preconditioner only;
the original double observations are re-factored on CPU in a well-conditioned
Cholesky-QR step. Singular or inaccurate preconditioners use Householder QR.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math

import torch

from openecon.analysis_contracts import AnalysisError


@dataclass
class ExecutionTrace:
    requested: str = "auto"
    operations: dict[str, int] = field(default_factory=dict)
    fallbacks: dict[str, int] = field(default_factory=dict)

    def record(self, device: str) -> None:
        self.operations[device] = self.operations.get(device, 0) + 1

    def fallback(self, reason: str) -> None:
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def metadata(self) -> dict:
        accelerated = [device for device in self.operations if device != "cpu"]
        return {"requested_device": self.requested,
                "device": accelerated[0] if len(accelerated) == 1 else "cpu",
                "factor_devices": dict(self.operations), "fallbacks": dict(self.fallbacks),
                "reporting_precision": "float64",
                "metal_precision": "float32 preconditioner, checked CPU float64 refinement"
                if "mps" in self.operations else None}


_TRACE: ContextVar[ExecutionTrace | None] = ContextVar("openecon_execution", default=None)


def validate_device(name: str) -> str:
    if name == "auto":
        return name
    try:
        device = torch.device(name)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_device", "Use auto, cpu, mps or an available CUDA device.") from exc
    if device.type == "cpu":
        return "cpu"
    if device.type == "mps" and device.index in (None, 0) and torch.backends.mps.is_available():
        return "mps"
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        if 0 <= index < torch.cuda.device_count():
            return str(device)
    raise AnalysisError("device_unavailable", "The requested tensor device is unavailable.")


@contextmanager
def execution_scope(device: str = "auto"):
    current = _TRACE.get()
    if current is not None and device == "auto":
        # An automatic helper shares its caller's actual device preference
        # and records all factors in the same end-to-end execution trace.
        yield current
        return
    trace = ExecutionTrace(validate_device(device))
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


def _preferred_device(trace: ExecutionTrace, block: torch.Tensor) -> str:
    if trace.requested != "auto":
        return trace.requested
    if torch.cuda.is_available():
        return "cuda"
    # Transfers/refinement outweigh acceleration for small or narrow blocks.
    if (torch.backends.mps.is_available() and block.shape[0] >= block.shape[1]
            and block.shape[0] * block.shape[1] ** 2 >= 8_000_000):
        return "mps"
    return "cpu"


def _metal_factor(block: torch.Tensor) -> torch.Tensor:
    n, k = block.shape
    if n < k:
        raise ArithmeticError("short_block")
    approximate = torch.linalg.qr(block.to(device="mps", dtype=torch.float32), mode="r").R
    # Copy before widening: Metal's R can be a strided view into the tall QR
    # workspace. A simultaneous device/dtype copy is not reliable for that
    # layout on all Torch releases.
    approximate = approximate.contiguous().cpu().double()
    singular = torch.linalg.svdvals(approximate)
    condition = float(singular[0] / singular[-1]) if float(singular[-1]) > 0 else math.inf
    if not math.isfinite(condition) or condition > 1e6:
        raise ArithmeticError("ill_conditioned_metal_preconditioner")
    # B = A R0^-1 uses the original double observations. Its Gram matrix is
    # close to identity; conditioning is that of B, not that of A squared.
    refined = torch.linalg.solve_triangular(approximate.T, block.T, upper=False).T
    gram = refined.T @ refined
    values = torch.linalg.eigvalsh(gram)
    if (not bool(torch.isfinite(values).all()) or float(values[0]) <= 0
            or float(values[-1] / values[0]) > 100):
        raise ArithmeticError("ill_conditioned_refinement")
    correction, info = torch.linalg.cholesky_ex((gram + gram.T) / 2)
    if int(info) != 0:
        raise ArithmeticError("refinement_not_positive_definite")
    upper = correction.T
    identity = torch.eye(k, dtype=torch.float64)
    inverse = torch.linalg.solve_triangular(upper, identity, upper=True)
    orthogonality = inverse.T @ gram @ inverse
    residual = block - refined @ approximate
    norm = float(torch.linalg.vector_norm(block))
    bound = 256 * torch.finfo(torch.float64).eps * max(1, k)
    if (float((orthogonality - identity).abs().max()) > bound
            or float(torch.linalg.vector_norm(residual)) > bound * max(norm, 1.0)):
        raise ArithmeticError("float64_refinement_certificate_failed")
    factor = upper @ approximate
    if not bool(torch.isfinite(factor).all()):
        raise ArithmeticError("non_finite_metal_factor")
    return factor


def qr_factor(block: torch.Tensor) -> torch.Tensor:
    """Return a CPU float64 R from one bounded CPU float64 observation block."""
    trace = _TRACE.get()
    if trace is None:
        return torch.linalg.qr(block, mode="r").R
    if block.device.type != "cpu" or block.dtype != torch.float64 or block.ndim != 2:
        raise AnalysisError("invalid_design", "Accelerated TSQR needs a CPU float64 block.")
    preferred = _preferred_device(trace, block)
    if preferred == "mps":
        try:
            result = _metal_factor(block)
            trace.record("mps")
            return result
        except (ArithmeticError, RuntimeError) as exc:
            trace.fallback(str(exc).splitlines()[0][:200])
    elif preferred.startswith("cuda"):
        try:
            result = torch.linalg.qr(block.to(preferred), mode="r").R.to("cpu")
            if bool(torch.isfinite(result).all()):
                trace.record(preferred)
                return result
            trace.fallback("non_finite_cuda_factor")
        except (RuntimeError, AssertionError) as exc:
            trace.fallback(str(exc).splitlines()[0][:200])
    trace.record("cpu")
    return torch.linalg.qr(block, mode="r").R
