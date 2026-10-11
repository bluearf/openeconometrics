"""Seeded wild-bootstrap multipliers and plus-one p-values shared by the wild routes.

Multiplier laws (mean 0, variance 1):

* ``rademacher`` -- +1 or -1 with probability 1/2 each
  (``torch.randint(2, ...) * 2 - 1``);
* ``webb`` -- Webb's six-point distribution, values
  +-sqrt(3/2), +-1, +-sqrt(1/2) with probability 1/6 each
  (``torch.randint(6, ...)`` indexing that ordered list);
* ``mammen`` -- (1 - sqrt 5)/2 with probability (sqrt 5 + 1)/(2 sqrt 5),
  otherwise (1 + sqrt 5)/2 (``torch.rand(...) < p``).

These are the conventions of ``wild_cluster_test``. Draws come from a private
``torch.Generator(cpu).manual_seed(seed)``; the matrix of B x G multipliers is
generated row-block by row-block in replication order, and CPU sampling is
serial, so the stream does not depend on the block size.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from numbers import Integral

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

KINDS = ("rademacher", "webb", "mammen")
_WEBB = torch.tensor([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)],
                     dtype=torch.float64)
_MAMMEN = ((1 - math.sqrt(5)) / 2, (1 + math.sqrt(5)) / 2, (math.sqrt(5) + 1) / (2 * math.sqrt(5)))


def check_count(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise AnalysisError("invalid_option", f"{name} must be an integer in {minimum}..{maximum}.")
    return int(value)


def check_kind(kind) -> str:
    if kind not in KINDS:
        raise AnalysisError("invalid_option", "wild must be 'rademacher', 'webb' or 'mammen'.")
    return kind


def draw(kind: str, rows: int, groups: int, generator: torch.Generator) -> Tensor:
    """[rows, groups] float64 multipliers of the declared law from ``generator``."""
    if kind == "rademacher":
        return (torch.randint(2, (rows, groups), generator=generator) * 2 - 1).to(torch.float64)
    if kind == "webb":
        return _WEBB[torch.randint(6, (rows, groups), generator=generator)]
    uniform = torch.rand((rows, groups), generator=generator, dtype=torch.float64)
    # Tensor operands keep float64; Python-scalar branches would give float32 values.
    lower = torch.tensor(_MAMMEN[0], dtype=torch.float64)
    upper = torch.tensor(_MAMMEN[1], dtype=torch.float64)
    return torch.where(uniform < _MAMMEN[2], lower, upper)


def blocks(kind: str, reps: int, groups: int, seed: int, block: int) -> Iterator[Tensor]:
    """Yield the B x G multiplier matrix in consecutive row blocks of at most ``block`` rows."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    done = 0
    while done < reps:
        rows = min(block, reps - done)
        yield draw(kind, rows, groups, generator)
        done += rows


def plus_one(exceed: int, reps: int) -> float:
    return (exceed + 1) / (reps + 1)


TIE_GUARD = 1e-10


def tail_p_values(observed: float, draws: Tensor, tail: str) -> float:
    """Plus-one bootstrap p-value of a signed statistic for the declared tail.

    Draws within a relative ``TIE_GUARD`` of the observed value count as
    exceedances: wild draws can reproduce the observed statistic exactly (for
    example equal multipliers in every cluster), and rounding must not decide
    such ties in the anti-conservative direction.
    """
    reps = draws.numel()
    slack = TIE_GUARD * max(1.0, abs(observed))
    if tail == "upper":
        return plus_one(int((draws >= observed - slack).sum()), reps)
    if tail == "lower":
        return plus_one(int((draws <= observed + slack).sum()), reps)
    if tail == "symmetric":
        return plus_one(int((draws.abs() >= abs(observed) - slack).sum()), reps)
    if tail == "equal_tail":
        upper = plus_one(int((draws >= observed - slack).sum()), reps)
        lower = plus_one(int((draws <= observed + slack).sum()), reps)
        return min(1.0, 2 * min(upper, lower))
    raise AnalysisError("invalid_option", "tail must be symmetric, equal_tail, upper or lower.")


def mc_error(probability: float, reps: int) -> float:
    return math.sqrt(probability * (1 - probability) / reps)
