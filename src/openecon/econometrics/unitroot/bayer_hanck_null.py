"""Null distributions of the Bayer-Hanck component tests (runtime asset loader).

``bayer_hanck_null.xz`` re-encodes ``NullDistr.dta`` of the authors' Stata
module ``bayerhanck`` (Christian Bayer and Christoph Hanck; SSC distribution
date 2024-10-06, http://fmwww.bc.edu/RePEc/bocode/n/NullDistr.dta, sha256
4975161431432e5441a96b029879a10a0efb1d536a7d61b262f562deb5c4e6bc; identical,
to float32 precision, to R/sysdata.rda ``null_dist`` of CRAN combcoint 0.2.0).
It holds 132 sorted columns of 10,000 simulated null draws: for deterministic
case c = 1 (none), 2 (constant), 3 (trend), number of variables N = 2..12 and
component j = 1 (Engle-Granger tau), 2 (Johansen lambda-max), 3 (Banerjee
et al. t), 4 (Boswijk Wald), column 44 (c - 1) + 4 (N - 2) + j.

Encoding (``scripts/generate_cointegration_breaks_oracles.py``): every draw is
rounded to a multiple of 1e-4 (the published draws are float32); the file is
two int32 (columns, draws), one int64 starting value per column and int32
increments, LZMA-compressed. Rounding changes a component p-value only when
the statistic lies within 5e-5 of a draw, and then by 1/10,000. The draws are
redistributed under combcoint's MIT license (docs/reference-licenses/combcoint-MIT.txt).
"""

from __future__ import annotations

import functools
import hashlib
import lzma
from pathlib import Path

import torch

from openecon.analysis_contracts import AnalysisError

ASSET = Path(__file__).with_name("bayer_hanck_null.xz")
# sha256 of the decompressed payload (independent of the LZMA build).
PAYLOAD_SHA256 = "0041f59794b6a67d963260412d46dc541e01e5b9dc476a84770a314ab87c1103"
QUANTUM = 1e-4
COMPONENTS = ("Engle-Granger", "Johansen", "Banerjee", "Boswijk")
CASES = ("none", "constant", "trend")
LOWER_TAIL = (True, False, True, False)        # EG and Banerjee reject for small values


@functools.lru_cache(maxsize=1)
def null_draws() -> torch.Tensor:
    """[132, 10000] float64 sorted null draws (multiples of 1e-4)."""
    try:
        packed = ASSET.read_bytes()
    except OSError as exc:
        raise AnalysisError("missing_calibration", "The Bayer-Hanck null distributions "
                            f"({ASSET.name}) are not installed.") from exc
    raw = bytearray(lzma.decompress(packed))
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
        raise AnalysisError("missing_calibration", "The Bayer-Hanck null-distribution asset "
                            "does not match its recorded checksum.")
    columns, draws = (int(v) for v in torch.frombuffer(raw, dtype=torch.int32, count=2))
    first = torch.frombuffer(raw, dtype=torch.int64, count=columns, offset=8)
    steps = torch.frombuffer(raw, dtype=torch.int32, count=columns * (draws - 1),
                             offset=8 + 8 * columns).reshape(columns, draws - 1)
    scaled = torch.cat([first[:, None], steps.to(torch.int64)], dim=1).cumsum(dim=1)
    return scaled.to(torch.float64) * QUANTUM


def component_p_value(statistic: float, component: int, n_variables: int, case: str) -> float:
    """Empirical p-value of the authors' procedure: tail share of the 10,000 draws + 1e-12.

    ``component`` 0..3 in ``COMPONENTS`` order; ``n_variables`` = 1 + regressors (2..12).
    """
    column = 44 * CASES.index(case) + 4 * (n_variables - 2) + component
    draws = null_draws()[column]
    value = torch.tensor([statistic], dtype=torch.float64)
    if LOWER_TAIL[component]:
        count = int(torch.searchsorted(draws, value, side="left"))
    else:
        count = draws.shape[0] - int(torch.searchsorted(draws, value, side="right"))
    return count / draws.shape[0] + 1e-12
