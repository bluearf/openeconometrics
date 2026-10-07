"""Small numerical contracts independent of clients and dataframe metadata."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


class KernelError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class KernelResult:
    parameters: Tensor
    covariance: Tensor
    fitted: Tensor
    log_likelihood: float
    condition_number: float
    iterations: int = 0
    solver: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
