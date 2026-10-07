"""Survivor and cumulative hazard curves after ``stcox``: Stata's ``stcurve``.

For covariate values ``x`` (Stata's default: the estimation-sample means; ``at``
overrides individual terms) the Cox model gives

    H(t | x) = H0_s(t) exp(x'b),     S(t | x) = exp(-H(t | x)),
    S_KP(t | x) = S0_KP,s(t)^exp(x'b),

with the Breslow baseline ``H0_s(t) = sum_{tau_k <= t} d_k / sum_{R_k} w exp(x_i'b + offset_i)``
of stratum s (Efron's averaged risk sets after ``ties='efron'``) and the
Kalbfleisch-Prentice baseline survivor ``S0_KP`` of Stata's ``predict basesurv``
(``CoxObjective.baseline``). Without ``data`` the curve is computed from the bounded baseline
table stored in the result (at most 400 failure times, evenly thinned); with
the estimation ``data`` the full step function at every failure time is
recomputed from the stored coefficients.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.models import ResultBundle


def _xb(result: ResultBundle, at: dict[str, float] | None) -> tuple[float, dict[str, float]]:
    means = dict(result.extra.get("covariate_means") or {})
    if any(term.startswith("tvc:") for term in (c.term for c in result.coefficients)):
        raise AnalysisError("unsupported_result", "stcurve is not available after tvc: the "
                            "curve depends on the path of the time-varying covariates.")
    values = dict(means)
    if at is not None:
        if not isinstance(at, dict):
            raise AnalysisError("invalid_option", "at must be a mapping of term names to values, "
                                "for example at={'age': 60, 'drug': 1}.")
        unknown = sorted(set(at) - set(values))
        if unknown:
            raise AnalysisError("invalid_option", f"at names unknown terms: {', '.join(unknown)}"
                                f" (terms: {', '.join(values)}).")
        for name, value in at.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise AnalysisError("invalid_option", f"at['{name}'] must be a finite number.")
            values[name] = float(value)
    estimates = {c.term: c.estimate for c in result.coefficients}
    return sum(estimates[name] * value for name, value in values.items()), values


def stcurve(result: ResultBundle, data: Any = None, at: dict[str, float] | None = None, *, batch_rows=None):
    """Survivor and cumulative hazard functions of a fitted Cox model (Stata ``stcurve``).

    Parameters
    ----------
    result : the ResultBundle of ``oe.stcox`` (not after ``tvc``; after ``ties='exactp'``
        the Peto-Breslow formulas are used at the exact estimates, as in Stata).
    data : the estimation data. When given, the full Breslow step function is
        recomputed at every failure time; otherwise the curve uses the bounded baseline
        table stored in ``result.extra['baseline']`` (at most 400 times).
    at : covariate values by term name (for categorical predictors the indicator terms,
        e.g. ``'group[2]'``); terms not named are set to their estimation-sample means, as
        Stata's ``stcurve`` does. ``at={term: 0 for every term}`` gives the baseline.

    Returns
    -------
    A table with columns ``stratum`` (stratified models), ``time``, ``cumulative_hazard``,
    ``survivor`` (``exp(-H)``) and ``survivor_kp`` (Kalbfleisch-Prentice, the estimator of
    Stata's ``basesurv``); ``attrs`` record the covariate values, ``xb`` and whether the
    curve was thinned.

    Stata: ``stcurve, survival at(age=60 drug=1)`` after ``stcox age drug``.

    Example
    -------
    >>> curve = oe.stcurve(cox_result, data=df, at={"drug": 1})   # doctest: +SKIP
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "stcox":
        raise AnalysisError("unsupported_result", "stcurve needs a result of oe.stcox.")
    xb, values = _xb(result, at)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .saved_prediction import cox_baseline
        from openecon.econometrics.postest.streaming_prediction import materialize_predictions
        baseline = cox_baseline(result, data, batch_rows=batch_rows)
        risk = math.exp(xb)
        def frames():
            for block in baseline.iter_batches(batch_rows=batch_rows or 65536):
                frame = block[["stratum", "time", "cumulative_hazard", "survivor_kp"]].copy()
                frame["cumulative_hazard"] *= risk
                frame["survivor"] = (-frame["cumulative_hazard"]).map(math.exp)
                frame["survivor_kp"] = frame["survivor_kp"].map(lambda x: x ** risk)
                if not result.extra.get("strata"):
                    frame = frame.drop(columns="stratum")
                yield frame
        return materialize_predictions(frames(), {"target": "full_stcurve", "thinned": False,
            "full_step_function": True, "at": values, "xb": xb,
            "source_hash": result.provenance["data_hash"]})
    if batch_rows is not None:
        raise AnalysisError("invalid_batch_size", "batch_rows applies to Dataset curves.")
    if data is None:
        baseline = result.extra.get("baseline")
        if not baseline:
            raise AnalysisError("unsupported_result", "This Cox result stores no baseline "
                                "function (models with tvc).")
        frame = pd.DataFrame({key: baseline[key] for key in
                              ("stratum", "time", "cumulative_hazard", "survivor_kp")})
        thinned = bool(baseline.get("thinned"))
    else:
        frame, thinned = _full_baseline(result, data), False
    risk = math.exp(xb)
    hazard = frame["cumulative_hazard"] * risk
    frame["cumulative_hazard"] = hazard
    frame["survivor"] = (-hazard).map(math.exp)
    # Kalbfleisch-Prentice: S(t | x) = S0_KP(t)^exp(x'b)
    frame["survivor_kp"] = frame.pop("survivor_kp").map(
        lambda value: value ** risk if value > 0 else 0.0)
    if not result.extra.get("strata"):
        frame = frame.drop(columns=["stratum"])
    return table(frame.reset_index(drop=True), at=values, xb=xb, thinned=thinned,
                 source="stored baseline table" if data is None else "recomputed from data")


def _full_baseline(result: ResultBundle, data: Any) -> pd.DataFrame:
    from openecon.econometrics.survival import cox, diagnostics
    from openecon.econometrics.survival.cox_kernels import CoxObjective

    spec = result.spec
    sample = cox.prepare(spec, data)
    if sample.terms != [c.term for c in result.coefficients]:
        raise AnalysisError("data_mismatch", "The data do not reproduce the model's terms; pass "
                            "the data the model was estimated on.")
    if sample.frame.positions != list(result.sample_positions):
        raise AnalysisError("data_mismatch", "The data do not reproduce the estimation sample; "
                            "pass the data the model was estimated on.")
    beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    ties = spec.options.get("ties", "breslow")
    # after exactp, predictions use the Peto-Breslow formulas (Stata's rule)
    objective = CoxObjective(sample.x, sample.risk, sample.weights.user
                             if sample.weights.weighted else None, sample.offset,
                             "efron" if ties == "efron" else "breslow")
    increments, log_alpha = objective.baseline(beta, float(sample.means @ beta))
    curve = diagnostics.baseline_table(sample.risk, increments, log_alpha, limit=None,
                                       labels=sample.strata_labels
                                       if sample.strata is not None else None)
    return pd.DataFrame({key: curve[key] for key in ("stratum", "time", "cumulative_hazard",
                                                     "survivor_kp")})
