"""Unequal-probability (PPS) without-replacement Horvitz–Thompson estimation.

Each sampled (first-stage) unit k has inclusion probability pi_k, design weight
d_k = 1/pi_k, and linearized value z_k = w_k e_k (``estimation`` module; z_k = y_k/pi_k
for an uncalibrated total), summed over the rows of a PSU. Strata are independent.
Within a stratum of n sampled units, with Delta-check D_kl = (pi_kl - pi_k pi_l)/pi_kl
(D_kk = 1 - pi_k), the variance estimators are (Särndal, Swensson and Wretman 1992,
sec. 2.8 and 3.6; R survey ``ppsvar``):

    ht     sum_k sum_l D_kl z_k z_l'                                 Horvitz–Thompson
    syg    -1/2 sum_k sum_l D_kl (z_k - z_l)(z_k - z_l)'             Sen–Yates–Grundy
           = ht - sum_k (sum_l D_kl) z_k z_k'

with exact joint probabilities (R ``ppsmat``) or the approximations of R survey:

    overton      pi_kl = 2(n-1) pi_k pi_l / (2n - pi_k - pi_l)                (Overton 1985)
    hartley_rao  pi_kl = (n-1) pi_k pi_l / (n - pi_k - pi_l + sum_U pi_j^2/n)  (Hartley and Rao 1962)

which give D_kl = alpha + beta_k + beta_l off the diagonal and are evaluated in O(n)
without the n-by-n matrix. Estimators needing only first-order probabilities:

    brewer          n/(n-1) sum_k (1 - pi_k)(z_k - zbar)(z_k - zbar)'   Brewer (2002) / Brewer and
                    Donadio (2003) with c_k = (n-1)/(n-pi_k); R survey pps="brewer"
    hajek           n/(n-1) sum_k (1 - pi_k)(z_k - A)(z_k - A)',  A = sum (1-pi_k) z_k / sum (1-pi_k)
                    (Hájek 1964; Berger 2004)
    hansen_hurwitz  n/(n-1) sum_k (z_k - zbar)(z_k - zbar)'             with-replacement approximation
                    (R survey probs= without fpc)

SYG and the approximations assume a fixed-size design within each stratum.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, tensor_bytes
from openecon.survey import _scalar

from .calibration import FLOAT, _key
from .estimation import calibration_record, estimate, frame_of, inference_table, options, resolve, targets

VARIANCES = ("ht", "syg", "brewer", "hajek", "hansen_hurwitz")
APPROXIMATIONS = ("overton", "hartley_rao")
MAX_JOINT_CELLS = 4_000_000


def _fail(message: str, code: str = "invalid_pps_option") -> AnalysisError:
    return AnalysisError(code, message)


class Geometry:
    """First-stage PSU codes, their probabilities and strata."""

    def __init__(self, frame: pd.DataFrame, prob, strata, psu):
        for role, column in (("prob", prob), ("strata", strata), ("psu", psu)):
            if column is not None and (not isinstance(column, str) or column not in frame.columns):
                raise AnalysisError("missing_columns", f"{role} must name a column of the data.")
            if column is not None and bool(frame[column].isna().any()):
                raise AnalysisError("invalid_pps_design", f"Design column '{column}' has missing values.")
        if prob is None:
            raise _fail("prob must name the inclusion-probability column.")
        if len({c for c in (prob, strata, psu) if c is not None}) != sum(c is not None for c in (prob, strata, psu)):
            raise _fail("prob, strata and psu must be distinct columns.")
        pi = _numeric(frame[prob], prob)
        if not bool(((pi > 0) & (pi <= 1)).all()):
            raise AnalysisError("invalid_pps_probability", "Inclusion probabilities must lie in (0, 1].")
        n = len(frame)
        self.strata_labels = list(pd.factorize(frame[strata], sort=False)[1]) if strata else [None]
        strata_rows = (torch.as_tensor(pd.factorize(frame[strata], sort=False)[0].astype("int64"))
                       if strata else torch.zeros(n, dtype=torch.int64))
        if psu is None:
            self.psu = torch.arange(n)
        else:
            keys = [strata, psu] if strata else [psu]
            self.psu = torch.as_tensor(frame.groupby(keys, sort=False).ngroup().to_numpy(dtype="int64", copy=True))
        groups = int(self.psu.max()) + 1
        high = torch.zeros(groups, dtype=FLOAT).scatter_reduce(0, self.psu, pi, "amax", include_self=False)
        low = torch.zeros(groups, dtype=FLOAT).scatter_reduce(0, self.psu, pi, "amin", include_self=False)
        if bool((high - low > 1e-12 * high).any()):
            raise AnalysisError("invalid_pps_probability", "Inclusion probabilities must be constant within each PSU.")
        self.pi = high
        self.owner = torch.zeros(groups, dtype=torch.int64).index_copy_(0, self.psu, strata_rows)
        self.counts = torch.bincount(self.owner)
        alone = (self.counts[self.owner] == 1) & (self.pi < 1)
        if bool(alone.any()):
            raise AnalysisError("singleton_pps_stratum", "A stratum with one sampled PSU (inclusion probability < 1) "
                                "has no variance estimate; collapse strata or declare certainty units with pi = 1.")
        self.d = 1.0 / pi
        self.prob, self.strata, self.psu_column = prob, strata, psu

    @property
    def df(self) -> int:
        return len(self.pi) - len(self.counts)

    def stratum_rows(self) -> list[Tensor]:
        return [torch.nonzero(self.owner == h).flatten() for h in range(len(self.counts))]


def _matrix(value, size: int, what: str) -> Tensor:
    if isinstance(value, pd.DataFrame) and value.shape != (size, size) or isinstance(value, (list, tuple)) and (
            len(value) != size or any(not hasattr(row, "__len__") or len(row) != size for row in value)):
        raise _fail(f"{what} must be a {size}-by-{size} matrix of joint inclusion probabilities.",
                    "pps_joint_probability")
    try:
        if isinstance(value, pd.DataFrame):
            value = value.to_numpy(dtype="float64")
        matrix = torch.as_tensor(value, dtype=FLOAT, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _fail(f"{what} must be a numeric square matrix.", "pps_joint_probability") from exc
    if matrix.shape != (size, size) or not bool(torch.isfinite(matrix).all()):
        raise _fail(f"{what} must be a finite {size}-by-{size} matrix.", "pps_joint_probability")
    return matrix


def _delta(matrix: Tensor, pi: Tensor, what: str) -> Tensor:
    """Validated Delta-check (pi_kl - pi_k pi_l)/pi_kl of one stratum."""
    scale = max(float(matrix.abs().max()), 1.0)
    if not torch.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12 * scale):
        raise _fail(f"{what} must be symmetric.", "pps_joint_probability")
    if not torch.allclose(matrix.diagonal(), pi, rtol=1e-9, atol=0.0):
        raise _fail(f"The diagonal of {what} must equal the declared inclusion probabilities.", "pps_joint_probability")
    product = pi[:, None] * pi[None, :]
    off = ~torch.eye(len(pi), dtype=torch.bool)
    smallest = torch.minimum(pi[:, None], pi[None, :])
    if (bool((matrix[off] <= 0).any()) or bool((matrix > smallest * (1 + 1e-9)).any())
            or bool((matrix < (pi[:, None] + pi[None, :] - 1) - 1e-9).any())):
        raise _fail(f"{what} needs 0 < pi_kl <= min(pi_k, pi_l) and pi_kl >= pi_k + pi_l - 1 for every sampled pair.",
                    "pps_joint_probability")
    delta = (matrix - product) / matrix
    delta.diagonal().copy_(1.0 - pi)
    return delta


def _blocks(joint, geometry: Geometry) -> list[tuple[Tensor, Tensor]]:
    """Per-stratum (PSU positions, Delta-check) from a full matrix or a stratum mapping."""
    strata = geometry.stratum_rows()
    sizes = [len(rows) for rows in strata]
    if isinstance(joint, Mapping):
        if sum(s * s for s in sizes) > MAX_JOINT_CELLS:
            raise AnalysisError("survey_budget", f"Joint probability blocks exceed {MAX_JOINT_CELLS:,} cells.")
        if geometry.strata is None:
            raise _fail("A mapping of joint probabilities needs strata=; pass one matrix for an unstratified design.")
        given = {_key(label, "Stratum labels"): matrix for label, matrix in joint.items()}
        keys = [_key(label, "Stratum labels") for label in geometry.strata_labels]
        if set(given) != set(keys) or len(given) != len(joint):
            raise _fail("joint must map every sampled stratum label (and only those) to its matrix.",
                        "pps_joint_probability")
        out = []
        for key, rows in zip(keys, strata):
            matrix = _matrix(given[key], len(rows), f"joint[{key[1]!r}]")
            out.append((rows, _delta(matrix, geometry.pi[rows], f"joint[{key[1]!r}]")))
        return out
    size = len(geometry.pi)
    if size * size > MAX_JOINT_CELLS:
        raise AnalysisError("survey_budget", f"A {size}-by-{size} joint probability matrix exceeds {MAX_JOINT_CELLS:,} "
                            "cells; supply per-stratum blocks as a mapping or use an approximation.")
    plan_workspace("PPS joint inclusion probabilities", {"joint matrix and Delta-check": tensor_bytes((size, size),
                                                                                                    itemsize=32)})
    matrix = _matrix(joint, size, "joint")
    product = geometry.pi[:, None] * geometry.pi[None, :]
    across = geometry.owner[:, None] != geometry.owner[None, :]
    if bool(((matrix - product).abs() > 1e-9 * product)[across].any()):
        raise _fail("Joint probabilities of units in different strata must equal pi_k pi_l (independent strata).",
                    "pps_joint_probability")
    return [(rows, _delta(matrix[rows][:, rows], geometry.pi[rows], "joint")) for rows in strata]


def _pi_squared(value, geometry: Geometry) -> Tensor:
    """Per-stratum sum_U pi_j^2 / n_h of the Hartley–Rao approximation."""
    counts = geometry.counts.to(FLOAT)
    sample = torch.zeros(len(counts), dtype=FLOAT).index_add_(0, geometry.owner, geometry.pi)
    if value is None:
        return sample / counts              # R survey HR(): the sample mean of pi estimates sum_U pi^2 / n
    squares = torch.zeros(len(counts), dtype=FLOAT).index_add_(0, geometry.owner, geometry.pi ** 2)
    if isinstance(value, Mapping):
        if geometry.strata is None:
            raise _fail("pi_squared_sum is one number for an unstratified design.")
        given = {_key(label, "Stratum labels"): v for label, v in value.items()}
        keys = [_key(label, "Stratum labels") for label in geometry.strata_labels]
        if set(given) != set(keys) or len(given) != len(value):
            raise _fail("pi_squared_sum must map every sampled stratum label (and only those) to sum_U pi^2.")
        raw = [given[k] for k in keys]
    elif len(counts) == 1:
        raw = [value]
    else:
        raise _fail("With strata, pi_squared_sum maps each stratum label to its population sum of pi^2.")
    raw = [_scalar(v) for v in raw]
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in raw):
        raise _fail("pi_squared_sum values must be finite numbers.")
    try:
        total = torch.tensor([float(v) for v in raw], dtype=FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _fail("pi_squared_sum values must be finite numbers.") from exc
    if (not bool(torch.isfinite(total).all()) or bool((total < squares * (1 - 1e-9)).any())
            or bool((total > counts * (1 + 1e-9)).any())):
        raise _fail("pi_squared_sum must lie between the sample sum of pi^2 and the stratum sample size "
                    "(sum_U pi_j^2 <= sum_U pi_j = n_h).")
    return total / counts


def pps_covariance(z: Tensor, geometry: Geometry, variance: str, joint, c: Tensor | None) -> Tensor:
    """Stratified without-replacement covariance of the PSU sums of z [n, k]."""
    k, strata = z.shape[1], len(geometry.counts)
    owner, pi = geometry.owner, geometry.pi
    totals = torch.zeros((len(pi), k), dtype=FLOAT).index_add_(0, geometry.psu, z)
    n = geometry.counts.to(FLOAT)
    active = (geometry.counts > 1)[owner]
    totals = totals * active[:, None]           # singleton strata are certainty units: no variance
    m = (n - 1).clamp(min=1.0)

    def stratum_sum(values):
        return torch.zeros((strata, *values.shape[1:]), dtype=FLOAT).index_add_(0, owner, values)

    if variance in ("brewer", "hajek", "hansen_hurwitz"):
        a = torch.ones_like(pi) if variance == "hansen_hurwitz" else 1.0 - pi
        if variance == "hajek":
            mass = stratum_sum(a)
            centre = stratum_sum(a[:, None] * totals) / torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
        else:
            centre = stratum_sum(totals) / n[:, None]
        centred = totals - centre[owner]
        coefficient = a * n[owner] / m[owner] * active
        return (centred * coefficient[:, None]).T @ centred
    if isinstance(joint, str):
        # D_kl = alpha_h + beta_k + beta_l (k != l), D_kk = 1 - pi_k.
        alpha = -1.0 / m if joint == "overton" else -(1.0 + c) / m
        beta = pi / (2.0 * m[owner]) if joint == "overton" else pi / m[owner]
        delta = 1.0 - pi
        s = stratum_sum(totals)
        b = stratum_sum(beta[:, None] * totals)
        diagonal = delta - alpha[owner] - 2.0 * beta
        covariance = ((totals * diagonal[:, None]).T @ totals + (s * alpha[:, None]).T @ s + b.T @ s + s.T @ b)
        if variance == "syg":
            rows = delta + (n[owner] - 1) * (alpha[owner] + beta) + stratum_sum(beta)[owner] - beta
            covariance = covariance - (totals * rows[:, None]).T @ totals
        return covariance
    covariance = torch.zeros((k, k), dtype=FLOAT)
    for rows, delta in joint:
        block = totals[rows]
        covariance += block.T @ delta @ block
        if variance == "syg":
            covariance -= (block * delta.sum(1)[:, None]).T @ block
    return covariance


def _pps(kind, data, outcomes, denominators, *, prob, strata, psu, variance, joint, pi_squared_sum, calibration,
         domain, missing, alpha, deff):
    alpha = options(alpha, missing, deff)
    if not isinstance(variance, str) or variance not in VARIANCES:
        raise _fail(f"variance must be one of {', '.join(VARIANCES)}.")
    exact = variance in ("ht", "syg")
    if exact and joint is None:
        raise _fail(f"variance='{variance}' needs joint inclusion probabilities (a matrix or per-stratum mapping) "
                    "or joint='overton'/'hartley_rao'.")
    if not exact and joint is not None:
        raise _fail(f"variance='{variance}' uses first-order probabilities only; joint applies to 'ht'/'syg'.")
    approximation = joint if isinstance(joint, str) else None
    if approximation is not None and approximation not in APPROXIMATIONS:
        raise _fail(f"joint must be a matrix, a per-stratum mapping or one of {', '.join(APPROXIMATIONS)}.")
    if pi_squared_sum is not None and approximation != "hartley_rao":
        raise _fail("pi_squared_sum is used only by joint='hartley_rao'.")
    frame = frame_of(data)
    geometry = Geometry(frame, prob, strata, psu)
    blocks = joint if joint is None or approximation else _blocks(joint, geometry)
    c = _pi_squared(pi_squared_sum, geometry) if approximation == "hartley_rao" else None
    fit = resolve(calibration, frame, geometry.d)
    t = targets(frame, kind, outcomes, denominators, domain, missing)
    theta, covariance, effects = estimate(t, geometry.d, fit, lambda z: pps_covariance(z, geometry, variance, blocks, c),
                                          deff)
    source = ("exact joint inclusion probabilities" if exact and not approximation
              else f"{approximation} approximation of pi_kl" if exact else "first-order inclusion probabilities only")
    if approximation == "hartley_rao":
        source += (" with supplied sum_U pi^2" if pi_squared_sum is not None
                   else " with sum_U pi^2 estimated by the sample sum of pi (R survey HR())")
    counts = geometry.counts.tolist()
    return inference_table(
        t, theta, covariance, geometry.df, alpha, effects, title=f"PPS without-replacement {kind}",
        variance=variance, joint=source,
        df_convention="all sampled PSUs minus strata, also for domains (R survey degf of the full sample)",
        calibration=calibration_record(fit),
        design={"prob": prob, "strata": strata, "psu": psu, "n_psu": len(geometry.pi), "n_strata": len(counts),
                "min_stratum_psus": min(counts), "max_stratum_psus": max(counts),
                "n_certainty_psus": int((geometry.pi == 1).sum()), "sum_design_weights": float(geometry.d.sum())},
        reference="Horvitz and Thompson (1952); Sen (1953); Yates and Grundy (1953); Hartley and Rao (1962); "
                  "Hájek (1964); Overton (1985); Brewer (2002); Brewer and Donadio (2003); R survey ppsvar()")


def survey_pps_total(data, outcomes, *, prob, strata=None, psu=None, variance="brewer", joint=None,
                     pi_squared_sum=None, calibration=None, domain=None, missing="raise", alpha=0.05, deff=False):
    """Horvitz–Thompson totals under PPS sampling without replacement with exact or approximate variances.

    The total sum_k y_k/pi_k (or the calibrated sum_k w_k y_k) is the R survey ``svytotal`` of a
    ``svydesign(fpc=~pi, pps=...)`` design; the variance follows ``variance``/``joint``.

    Parameters
    ----------
    data : table with one row per sampled element (every sampled PSU retained).
    outcomes : column name or list of 1..32 numeric columns.
    prob : column of first-stage inclusion probabilities pi_k in (0, 1], constant within PSU.
    strata : optional stratum column (independent fixed-size selections).
    psu : optional first-stage unit column; rows are summed to PSU totals. Without it
        every row is a sampled unit.
    variance : 'ht' or 'syg' (with ``joint``), 'brewer' (default), 'hajek' or
        'hansen_hurwitz' (with-replacement approximation).
    joint : for 'ht'/'syg': a square matrix of pi_kl over PSUs in first-appearance
        order (cross-stratum entries pi_k pi_l), a mapping stratum -> matrix of that
        stratum's PSUs, or the approximation 'overton' / 'hartley_rao'. At most
        4,000,000 matrix cells.
    pi_squared_sum : Hartley-Rao population sum_U pi_j^2 (a number, or a mapping per
        stratum); default: the sample sum of pi, as R survey's ``HR()``.
    calibration : optional svycal result/specification; the design weights are 1/pi.
    domain : optional complete Boolean/0-1 domain column.
    missing : 'raise' (default) or 'drop' in-domain rows with missing target values.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE`` comparator.

    Returns
    -------
    Table with estimate, std_error, statistic, p_value, ci_low, ci_high (and deff) per
    target; ``attrs`` holds the full covariance, df (sampled PSUs minus strata), the
    variance method, joint-probability source, sample counts and calibration record.
    """
    return _pps("total", data, outcomes, None, prob=prob, strata=strata, psu=psu, variance=variance, joint=joint,
                pi_squared_sum=pi_squared_sum, calibration=calibration, domain=domain, missing=missing, alpha=alpha,
                deff=deff)


def survey_pps_mean(data, outcomes, *, prob, strata=None, psu=None, variance="brewer", joint=None,
                    pi_squared_sum=None, calibration=None, domain=None, missing="raise", alpha=0.05, deff=False):
    """Hájek means sum(y/pi)/sum(1/pi) under PPS sampling without replacement with linearized variances.

    The influence (y_k - mean)/N^ times 1/pi_k enters the chosen variance estimator
    (R survey ``svymean`` of a PPS design).

    Parameters
    ----------
    data : table with one row per sampled element (every sampled PSU retained).
    outcomes : column name or list of 1..32 numeric columns.
    prob : column of first-stage inclusion probabilities pi_k in (0, 1], constant within PSU.
    strata : optional stratum column (independent fixed-size selections).
    psu : optional first-stage unit column; rows are summed to PSU totals. Without it
        every row is a sampled unit.
    variance : 'ht' or 'syg' (with ``joint``), 'brewer' (default), 'hajek' or
        'hansen_hurwitz' (with-replacement approximation).
    joint : for 'ht'/'syg': a square matrix of pi_kl over PSUs in first-appearance
        order (cross-stratum entries pi_k pi_l), a mapping stratum -> matrix of that
        stratum's PSUs, or the approximation 'overton' / 'hartley_rao'. At most
        4,000,000 matrix cells.
    pi_squared_sum : Hartley-Rao population sum_U pi_j^2 (a number, or a mapping per
        stratum); default: the sample sum of pi, as R survey's ``HR()``.
    calibration : optional svycal result/specification; the design weights are 1/pi.
    domain : optional complete Boolean/0-1 domain column.
    missing : 'raise' (default) or 'drop' in-domain rows with missing target values.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE`` comparator.

    Returns
    -------
    Table with estimate, std_error, statistic, p_value, ci_low, ci_high (and deff) per
    target; ``attrs`` holds the full covariance, df (sampled PSUs minus strata), the
    variance method, joint-probability source, sample counts and calibration record.
    """
    return _pps("mean", data, outcomes, None, prob=prob, strata=strata, psu=psu, variance=variance, joint=joint,
                pi_squared_sum=pi_squared_sum, calibration=calibration, domain=domain, missing=missing, alpha=alpha,
                deff=deff)


def survey_pps_ratio(data, numerators, denominators, *, prob, strata=None, psu=None, variance="brewer", joint=None,
                     pi_squared_sum=None, calibration=None, domain=None, missing="raise", alpha=0.05, deff=False):
    """Ratios of Horvitz–Thompson totals under PPS sampling without replacement with linearized variances.

    The influence (y_k - R x_k)/X^ times 1/pi_k enters the chosen variance estimator
    (R survey ``svyratio`` of a PPS design).

    Parameters
    ----------
    data : table with one row per sampled element (every sampled PSU retained).
    numerators, denominators : paired lists of numeric columns (one denominator each).
    prob : column of first-stage inclusion probabilities pi_k in (0, 1], constant within PSU.
    strata : optional stratum column (independent fixed-size selections).
    psu : optional first-stage unit column; rows are summed to PSU totals. Without it
        every row is a sampled unit.
    variance : 'ht' or 'syg' (with ``joint``), 'brewer' (default), 'hajek' or
        'hansen_hurwitz' (with-replacement approximation).
    joint : for 'ht'/'syg': a square matrix of pi_kl over PSUs in first-appearance
        order (cross-stratum entries pi_k pi_l), a mapping stratum -> matrix of that
        stratum's PSUs, or the approximation 'overton' / 'hartley_rao'. At most
        4,000,000 matrix cells.
    pi_squared_sum : Hartley-Rao population sum_U pi_j^2 (a number, or a mapping per
        stratum); default: the sample sum of pi, as R survey's ``HR()``.
    calibration : optional svycal result/specification; the design weights are 1/pi.
    domain : optional complete Boolean/0-1 domain column.
    missing : 'raise' (default) or 'drop' in-domain rows with missing target values.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE`` comparator.

    Returns
    -------
    Table with estimate, std_error, statistic, p_value, ci_low, ci_high (and deff) per
    target; ``attrs`` holds the full covariance, df (sampled PSUs minus strata), the
    variance method, joint-probability source, sample counts and calibration record.
    """
    return _pps("ratio", data, numerators, denominators, prob=prob, strata=strata, psu=psu, variance=variance,
                joint=joint, pi_squared_sum=pi_squared_sum, calibration=calibration, domain=domain, missing=missing,
                alpha=alpha, deff=deff)
