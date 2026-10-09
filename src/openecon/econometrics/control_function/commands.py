"""Public bounded continuous-first-stage conditional-mean control functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis import _coerce_frame, fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import tensor_bytes

from . import KINDS
from .kernels import fit_joint
from .state import capture_state, typed_cluster_codes


def _convenience(name, data, y, endogenous, x, instruments, covariance, cluster,
                 intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows):
    spec = make_spec(name, outcome=y, predictors=column_list(x, "x"),
                     columns={"endogenous": endogenous, "instruments": column_list(instruments, "instruments")},
                     covariance=covariance, cluster=cluster, intercept=intercept,
                     missing=missing, alpha=alpha,
                     options={"max_iterations": max_iterations, "tolerance": tolerance,
                              "max_work": max_work, "device": device, "batch_rows": batch_rows})
    return fit(spec, data=data)


def fit_control_function(spec: ModelSpec, data: Any) -> ResultBundle:
    """Fit both stages and report their full joint coefficient covariance."""
    with torch.device("cpu"):
        return _fit(spec, data)


def _fit(spec, data):
    from openecon.dataset import Dataset

    if spec.options.get("device", "cpu") != "cpu":
        raise AnalysisError("unsupported_device", "Control functions support explicit CPU float64 only.")
    if isinstance(data, Dataset):
        return fit_stream_control_function(spec, data)
    data = _coerce_frame(data)
    if len(data) > 5000 or "batch_rows" in spec.options:
        return fit_stream_control_function(spec, Dataset.from_frame(data))
    endogenous = spec.columns["endogenous"]
    instruments = spec.columns["instruments"]
    exogenous = list(spec.predictors)
    if len(exogenous) > 16 or len(instruments) > 16:
        raise AnalysisError("dimension_limit", "Control functions admit at most 16 exogenous predictors and 16 excluded instruments.")
    names = [spec.outcome, endogenous, *exogenous, *instruments]
    if len(set(names)) != len(names) or not instruments:
        raise AnalysisError("invalid_spec", "Outcome, endogenous, exogenous and excluded-instrument columns must be distinct; instruments cannot be empty.")
    frame = ModelFrame(spec, data)
    width_z = len(exogenous) + len(instruments) + int(spec.intercept)
    width_q = len(exogenous) + 2 + int(spec.intercept)
    joint_width = width_z + width_q
    if joint_width > 36 or frame.n <= joint_width:
        raise AnalysisError("dimension_limit", "Control functions need at most 36 joint coefficients and more retained rows than coefficients.")
    frame.workspace_plan("control-function joint score, covariance and complete saved state", {
        "input_designs_scores_and_saved_state": tensor_bytes((frame.n, joint_width + 12), itemsize=192),
        "joint_bread_meat_covariance_and_restore_buffers": tensor_bytes((joint_width, joint_width), itemsize=192),
        "cluster_score_and_sample_buffers": tensor_bytes((frame.n, joint_width + 6), itemsize=24),
    })
    zdesign = frame.design([*exogenous, *instruments])
    xdesign = frame.design([*exogenous, endogenous])
    d, y = frame.numeric(endogenous), frame.numeric(spec.outcome)
    if bool(((d == 0) | (d == 1)).all()):
        raise AnalysisError("unsupported_endogenous", "Binary endogenous first stages are outside this continuous-first-stage contract.")
    cluster_name = spec.cluster[0] if isinstance(spec.cluster, list) else spec.cluster
    clusters, count = (typed_cluster_codes(frame.sample[cluster_name]) if spec.covariance == "cluster" else (None, None))
    kind = KINDS[spec.estimator][0]
    solved = kernel_call(fit_joint, zdesign.x, xdesign.x, d, y, kind,
                         cluster_codes=clusters, max_iterations=frame.option("max_iterations"),
                         tolerance=frame.option("tolerance"), max_work=frame.option("max_work"))
    state = capture_state(frame, zdesign, xdesign, d, y, solved, kind,
                          clusters=clusters, original_index=data.index)
    terms = [f"first_stage:{term}" for term in zdesign.terms]
    terms += [f"outcome:{term}" for term in [*xdesign.terms, "ControlResidual"]]
    params = torch.cat((solved["gamma"], solved["beta"]))
    covariance = solved["joint_covariance"]
    endogeneity = wald_test(params, covariance, [joint_width - 1])
    return build_result(
        frame, terms=terms, params=params, covariance=covariance,
        equations=["first_stage"] * width_z + ["outcome"] * width_q,
        fitted=solved["fitted"], observed=y, use_t=False,
        metrics={"working_criterion": solved["criterion"], "first_stage_n_parameters": width_z,
                 "outcome_n_parameters": width_q},
        tests={"control_coefficient_zero": endogeneity},
        solver="OLS first stage and existing GLM conditional-mean kernel",
        optimizer=solved["optimizer"],
        inference={"cluster_count": count, "small_sample_correction": 1.0,
                   "correction": "full stacked-equation HC0" if clusters is None else "full stacked-equation CR0",
                   "generated_control_uncertainty": True,
                   "distribution": "normal", "use_t": False},
        extra={"control_function_state": state},
        provenance={"control_function_state_sha256": state["integrity_sha256"]},
        warnings=["Conditional mean/control sufficiency and instrument exclusion are caller assumptions; inference is asymptotic.",
                  "No weak-IV, finite-cluster, structural treatment-effect or response-distribution guarantee."],
    )


def fit_stream_control_function(spec: ModelSpec, data: Any) -> ResultBundle:
    """Global two-stage Dataset fit with bounded buffers and source-bound state."""
    from openecon.econometrics.streaming_control_function import solve_stream
    from .kernels import _family_link
    from .stream_state import bind_source, capture_stream_state

    with torch.no_grad(), torch.device("cpu"):
        sample, solved = kernel_call(solve_stream, spec, data,
                                     batch_rows=spec.options.get("batch_rows"))
        state = capture_stream_state(sample, solved, KINDS[spec.estimator][0])
        zterms, xterms = state["z_terms"], state["x_terms"]
        frame = ModelFrame(spec, sample.sample)
        zdesign = frame.design([*spec.predictors, *spec.columns["instruments"]])
        xdesign = frame.design([*spec.predictors, spec.columns["endogenous"]])
        residual = frame.numeric(spec.columns["endogenous"]) - zdesign.x @ solved["gamma"]
        q = torch.cat((xdesign.x, residual[:, None]), dim=1)
        _, link_type = _family_link(KINDS[spec.estimator][0])
        fitted = link_type().inverse(q @ solved["beta"])
        terms = [f"first_stage:{term}" for term in zterms]
        terms += [f"outcome:{term}" for term in [*xterms, "ControlResidual"]]
        params = torch.cat((solved["gamma"], solved["beta"]))
        covariance = solved["joint_covariance"]
        joint_width, width_z, width_q = len(terms), len(zterms), len(xterms) + 1
        result = build_result(
            frame, terms=terms, params=params, covariance=covariance,
            equations=["first_stage"] * width_z + ["outcome"] * width_q,
            fitted=fitted, observed=frame.numeric(spec.outcome), use_t=False, nobs=sample.nobs,
            metrics={"working_criterion": solved["criterion"],
                     "first_stage_n_parameters": width_z, "outcome_n_parameters": width_q},
            tests={"control_coefficient_zero": wald_test(params, covariance, [joint_width - 1])},
            solver="global replayed TSQR first stage and analytic conditional-mean equations",
            optimizer=solved["optimizer"],
            inference={"cluster_count": state["cluster_count"], "small_sample_correction": 1.0,
                       "correction": "full stacked-equation HC0" if spec.covariance == "robust"
                       else "full stacked-equation CR0", "generated_control_uncertainty": True,
                       "distribution": "normal", "use_t": False},
            extra={"control_function_state": state},
            provenance={**sample.provenance(),
                        "execution": solved["execution"], "device": solved["execution"]["device"],
                        "control_function_state_sha256": state["integrity_sha256"],
                        "prediction_sample": "first400 retained observations",
                        "control_function_state": "compact; unchanged estimation source required for semantic replay"},
            warnings=["Conditional mean/control sufficiency and instrument exclusion are caller assumptions; inference is asymptotic.",
                      "No weak-IV, finite-cluster, structural treatment-effect or response-distribution guarantee.",
                      "Compact saved models require the unchanged estimation Dataset for semantic replay."],
        )
        for row in result.predictions:
            row["row"] = sample.sample_positions[int(row["row"])]
        result.nobs_original = sample.original_count
        result.dropped_rows = sample.original_count - sample.nrows
        if result.dropped_rows:
            result.warnings.append(f"Excluded {result.dropped_rows} observation(s) with missing model inputs.")
        result.sample_positions = []
        return bind_source(result, data)


def cfregress(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
              covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
              missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
              tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Gaussian identity mean with estimated-control HC0 or whole-cluster CR0."""
    return _convenience("cfregress", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cflogit(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
            covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
            missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
            tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Binary logistic conditional mean with full two-stage HC0 or CR0."""
    return _convenience("cflogit", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cfprobit(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
             covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
             missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
             tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Binary conditional-scale probit with full estimated-control covariance."""
    return _convenience("cfprobit", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cfcloglog(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
              covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
              missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
              tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Binary complementary-log-log control function with stacked HC0/CR0."""
    return _convenience("cfcloglog", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cfpoisson(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
              covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
              missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
              tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Integer-response log mean with full generated-control HC0/CR0."""
    return _convenience("cfpoisson", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cfgamma(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
            covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
            missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
            tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Positive Gamma log mean-score; fixed working dispersion, stacked HC0/CR0."""
    return _convenience("cfgamma", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cfinvgauss(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
               covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
               missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
               tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Positive inverse-Gaussian log mean-score with observed stacked HC0/CR0."""
    return _convenience("cfinvgauss", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)


def cffraclogit(*, data: Any, y: str, endogenous: str, instruments: Sequence[str], x: Sequence[str] | None = None,
                covariance: str = "robust", cluster: str | None = None, intercept: bool = True,
                missing: str = "raise", alpha: float = 0.05, max_iterations: int = 100,
                tolerance: float = 1e-9, max_work: int = 10000000000, device: str = "cpu", batch_rows: int | None = None) -> ResultBundle:
    """Fractional [0,1] logit quasi-score with full first-stage HC0/CR0."""
    return _convenience("cffraclogit", data, y, endogenous, x, instruments, covariance, cluster, intercept, missing, alpha, max_iterations, tolerance, max_work, device, batch_rows)
