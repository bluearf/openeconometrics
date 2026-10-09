"""Exact binary-mediator standardization with complete joint saved inference.

The four means average over the fixed retained empirical control distribution.
No mediator sampling, outcome simulation, or restoration refit is involved.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.resources import plan_workspace
from .mediation import CAUSAL_ASSUMPTIONS
from .binary_kernels import evaluate_joint, fit_joint

DT = torch.float64
MEANS = ["mu00", "mu10", "mu01", "mu11"]
EFFECTS = ["PNDE", "TNDE", "PNIE", "TNIE", "TE"]
CONTRAST = torch.tensor(
    [[-1, 1, 0, 0], [0, 0, -1, 1], [-1, 0, 1, 0], [0, -1, 0, 1], [-1, 0, 0, 1]],
    dtype=DT, device="cpu",
)
COUNTERFACTUAL_COLUMNS = [
    "mediator_probability0", "mediator_probability1", "outcome_mean00", "outcome_mean10",
    "outcome_mean01", "outcome_mean11", *MEANS,
    "mediator_fitted_probability", "outcome_fitted_mean",
]
NOTES = [
    "Associational standardization is the default; causal interpretation additionally requires the declared identification assumptions.",
    "Uncertainty conditions on the retained empirical control distribution; it does not include sampling variation of that distribution.",
    "TE=PNDE+TNIE=TNDE+PNIE. The five-effect covariance is structurally singular; no full-rank global Wald test is reported.",
    "Both equations are fitted jointly for score-based covariance; HC0 retains the cross-equation score blocks without a degrees-of-freedom multiplier.",
]


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < level < 1:
        _error("level must be a finite probability strictly between zero and one.", "invalid_option")
    level = float(level)
    z = float(torch.special.ndtri(torch.tensor((1 + level) / 2, dtype=DT, device="cpu")))
    if not math.isfinite(z):
        _error("level is too close to one for float64 normal inference.", "invalid_option")
    return level, z


def _options(mediator_link, outcome_model, interaction, covariance, interpretation, assumptions,
             max_iterations, tolerance):
    if mediator_link not in ("logit", "probit") or not isinstance(mediator_link, str):
        _error("mediator_link must be logit or probit.", "invalid_option")
    if outcome_model not in ("gaussian", "logit", "probit", "poisson") or not isinstance(outcome_model, str):
        _error("outcome_model must be gaussian, logit, probit or poisson.", "invalid_option")
    if not isinstance(interaction, bool):
        _error("interaction must be True or False.", "invalid_option")
    if covariance not in ("OIM", "HC0") or not isinstance(covariance, str):
        _error("covariance must be OIM or HC0.", "invalid_option")
    if interpretation not in ("associational", "causal") or not isinstance(interpretation, str):
        _error("interpretation must be associational or causal.", "invalid_option")
    if assumptions is None:
        assumptions = []
    if not isinstance(assumptions, (list, tuple, set, frozenset)) or any(
        not isinstance(value, str) for value in assumptions
    ) or len(set(assumptions)) != len(assumptions) or set(assumptions) - CAUSAL_ASSUMPTIONS:
        _error("assumptions must contain distinct supported causal assumption names.", "invalid_option")
    assumptions = sorted(assumptions)
    if interpretation == "causal" and not CAUSAL_ASSUMPTIONS.issubset(assumptions):
        _error("Causal interpretation requires consistency, positivity, sequential_ignorability and no_exposure_induced_mediator_outcome_confounding declarations.", "causal_assumptions_required")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or not 1 <= max_iterations <= 1000:
        _error("max_iterations must be an integer in [1,1000].", "invalid_option")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 1e-12 <= tolerance <= 1e-5:
        _error("tolerance must be finite and in [1e-12,1e-5].", "invalid_option")
    return dict(mediator_link=mediator_link, outcome_model=outcome_model, interaction=interaction,
                covariance=covariance, interpretation=interpretation, assumptions=assumptions,
                max_iterations=int(max_iterations), tolerance=float(tolerance))


def _column_names(y, treatment, mediator, controls):
    if controls is None:
        controls = []
    elif isinstance(controls, str):
        controls = [controls]
    elif isinstance(controls, (list, tuple)):
        controls = list(controls)
    else:
        _error("controls must be a column name or a list of distinct column names.", "invalid_spec")
    names = [y, treatment, mediator, *controls]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        _error("Outcome, treatment, mediator and control names must be distinct nonempty strings.", "invalid_spec")
    if len(controls) > 10:
        _error("At most ten controls are supported.", "resource_limit")
    return dict(y=y, treatment=treatment, mediator=mediator, controls=controls)


def _plan(n, k, options, *, budget_bytes=None):
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= 4096:
        _error("Binary mediation supports 1..4096 retained observations.", "resource_limit")
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= 10:
        _error("Binary mediation supports at most ten controls.", "resource_limit")
    q = 2 + k
    r = 3 + int(options["interaction"]) + k + int(options["outcome_model"] == "gaussian")
    p = q + r
    work = n * p*p * options["max_iterations"]
    if work > 2_000_000_000:
        _error("The declared observation/parameter/iteration work exceeds 2e9 cells.", "resource_limit")
    return plan_workspace("binary_mediation", {
        "inputs_and_designs": 8*n*(4*k + 30),
        "score_and_autodiff_buffers": 8*(8*n*p*p + 12*n*p + 32*p*p),
        "complete_standardization": 8*(30*n + 18*p + 81),
    }, budget_bytes=budget_bytes).record() | {"estimated_iteration_cells": work, "max_iteration_cells": 2_000_000_000}


def _vector(value, n, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad or value.is_complex() or value.ndim != 1:
            _error(f"{name} requires a real one-dimensional CPU vector without gradients.", "unsupported_input")
        values = value.tolist()
    elif isinstance(value, pd.Series) or type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", None) != 1:
            _error(f"{name} requires a one-dimensional vector.", "unsupported_input")
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        _error(f"{name} requires a resident numeric vector.", "unsupported_input")
    if len(values) != n:
        _error("Selected input vectors must have equal lengths.")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        _error(f"{name} requires real numeric cells without booleans or missing values.")
    if any(isinstance(v, Integral) and abs(v) > 1e12 for v in values):
        _error(f"{name} magnitudes exceed the supported 1e12 bound.")
    if any(not math.isfinite(float(v)) for v in values):
        _error(f"{name} contains missing or nonfinite cells; rows cannot be silently dropped.", "missing_values")
    if any(abs(v) > 1e12 for v in values):
        _error(f"{name} magnitudes exceed the supported 1e12 bound.")
    return torch.tensor(values, dtype=DT, device="cpu")


def _binary(value, name):
    if set(value.tolist()) != {0.0, 1.0}:
        _error(f"{name} must contain exactly numeric 0 and 1, with both levels observed; no recoding.", "invalid_binary")


def _inputs(data, columns, options):
    names = [columns["y"], columns["treatment"], columns["mediator"], *columns["controls"]]
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            _error("Input column labels must be unique.", "invalid_spec")
        if any(name not in data.columns for name in names):
            _error("Every selected column must be present in data.", "invalid_spec")
        n = len(data)
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Every selected column must be present in data.", "invalid_spec")
        try:
            n = len(data[names[0]])
        except TypeError as error:
            raise AnalysisError("unsupported_input", "Data columns require resident positional vectors.") from error
    else:
        _error("Use a resident DataFrame or column mapping; Dataset and implicit collection are unsupported.", "unsupported_input")
    plan = _plan(n, len(columns["controls"]), options)
    y, a, m, *control = [_vector(data[name], n, name) for name in names]
    c = torch.stack(control, dim=1) if control else torch.empty((n, 0), dtype=DT, device="cpu")
    _binary(a, "treatment")
    _binary(m, "mediator")
    if options["outcome_model"] in ("logit", "probit"):
        _binary(y, "outcome")
    elif options["outcome_model"] == "poisson" and bool(((y < 0) | (y != y.floor())).any()):
        _error("Poisson outcome requires nonnegative integers; no rounding or recoding.", "invalid_count")
    return y, m, a, c, plan


def _parameter_names(columns, options):
    names = ["mediator:_cons", "mediator:" + columns["treatment"],
            *["mediator:" + name for name in columns["controls"]],
            "outcome:_cons", "outcome:" + columns["treatment"], "outcome:" + columns["mediator"],
            *(["outcome:" + columns["treatment"] + "*" + columns["mediator"]] if options["interaction"] else []),
            *["outcome:" + name for name in columns["controls"]],
            *(["outcome:log_sigma"] if options["outcome_model"] == "gaussian" else [])]
    if len(set(names)) != len(names):
        _error("Column names collide with generated intercept/interaction/scale parameter labels.", "invalid_spec")
    return names


def _link(eta, kind):
    if kind == "logit":
        return torch.sigmoid(eta)
    if kind == "probit":
        return torch.special.ndtr(eta)
    if kind == "poisson":
        return eta.exp()
    return eta


def _row_targets(theta, y, m, a, c, options):
    n, k = c.shape
    cut = 2 + k
    bm, by = theta[:cut], theta[cut:]
    mediator0 = bm[0] + c @ bm[2:]
    probabilities = [_link(mediator0 + av*bm[1], options["mediator_link"]) for av in (0, 1)]
    control_start = 3 + int(options["interaction"])
    control_y = c @ by[control_start:control_start+k]
    outcomes = []
    for av, mv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        eta = by[0] + av*by[1] + mv*by[2] + control_y
        if options["interaction"]:
            eta = eta + av*mv*by[3]
        outcomes.append(_link(eta, options["outcome_model"]))
    row_means = torch.stack([
        (1-probabilities[ap])*outcomes[av] + probabilities[ap]*outcomes[2+av]
        for av, ap in ((0, 0), (1, 0), (0, 1), (1, 1))
    ], dim=1)
    observed_eta = by[0] + a*by[1] + m*by[2] + control_y
    if options["interaction"]:
        observed_eta = observed_eta + a*m*by[3]
    observed_m = _link(mediator0+a*bm[1], options["mediator_link"])
    rows = torch.cat([torch.stack(probabilities+outcomes, dim=1), row_means,
                      observed_m[:, None], _link(observed_eta, options["outcome_model"])[:, None]], dim=1)
    if not bool(torch.isfinite(rows).all()):
        _error("Counterfactual model means exceed finite float64 support.", "numerical_failure")
    return row_means, rows


def _sym(value):
    return (value + value.T) / 2


def _derived(theta, covariance, y, m, a, c, options):
    def targets(value):
        means = _row_targets(value, y, m, a, c, options)[0].mean(dim=0)
        return torch.cat([means, CONTRAST @ means])
    values = targets(theta)
    jacobian = torch.autograd.functional.jacobian(targets, theta)
    joint = _sym(jacobian @ covariance @ jacobian.T)
    rows = _row_targets(theta, y, m, a, c, options)[1]
    if not bool(torch.isfinite(jacobian).all() & torch.isfinite(joint).all()):
        _error("Standardized mean/effect covariance is not finite.", "numerical_failure")
    scale = float(joint.abs().max())
    if float(torch.linalg.eigvalsh(joint)[0]) < -1e-10*max(scale, 1e-300):
        _error("Standardized target covariance failed its positive-semidefinite gate.", "invalid_covariance")
    natural = theta.clone()
    derivative = torch.ones_like(theta)
    if options["outcome_model"] == "gaussian":
        natural[-1] = theta[-1].exp()
        derivative[-1] = natural[-1]
    natural_covariance = _sym(covariance*derivative[:, None]*derivative[None, :])
    return dict(values=values.tolist(), jacobian=jacobian.tolist(), covariance=joint.tolist(),
                counterfactual_rows=rows.tolist(), natural_parameters=natural.tolist(),
                natural_covariance=natural_covariance.tolist())


def _checksum(state):
    return hashlib.sha256(json.dumps({key: value for key, value in state.items() if key != "checksum"},
                                    sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _settings(columns, options, n, plan, level):
    return dict(columns=columns, **options, level=level, n=n, n_input=n, n_retained=n,
                n_dropped=0, device="cpu", precision="float64", missing="raise", weights=None,
                cluster=None, treatment0=0, treatment1=1, binary_coding="exact numeric 0/1; no recoding",
                numeric_magnitude_bound=1e12, max_observations=4096, max_controls=10,
                sample="resident positional complete iid rows", controls_distribution="fixed retained empirical controls",
                target="exact two-state binary-mediator standardized outcome means and natural-effect contrasts",
                mean_order=MEANS, effect_order=EFFECTS, effect_scale="outcome units; probability differences for binary outcomes",
                inference="joint equation OIM or uncorrected HC0, full delta covariance, asymptotic normal zero-null effect inference",
                inference_df=None, effect_covariance_rank_bound=3, global_effect_wald=None,
                restoration="complete saved state replay; no optimizer and no refit",
                complete_inputs_saved=True, complete_joint_covariance_saved=True, resources=plan,
                input_position_column=_input_position_column(columns),
                excluded_scope=["Dataset", "weights", "clusters", "categorical encoding", "mediator simulation",
                                "new-data controls", "empirical-control-distribution sampling variance"])


def _matrix(value, names):
    return table([[name, *row] for name, row in zip(names, value)], columns=["parameter", *names])


def _input_position_column(columns):
    names = [columns["y"], columns["treatment"], columns["mediator"], *columns["controls"]]
    marker = "row"
    while marker in names:
        marker = "_" + marker
    return marker


def _inference_rows(names, estimates, covariance, level, *, label, nuisance=None, test_zero=True):
    _, zcrit = _confidence(level)
    rows = []
    for i, (name, estimate) in enumerate(zip(names, estimates)):
        variance = covariance[i][i]
        if variance < 0:
            if abs(variance) > 1e-12*max(max(abs(v) for row in covariance for v in row), 1e-300):
                _error("A saved target has negative variance.", "invalid_covariance")
            variance = 0.0
        se = math.sqrt(variance)
        stat = estimate/se if se > 0 and name != nuisance and test_zero else None
        pvalue = math.erfc(abs(stat)/math.sqrt(2)) if stat is not None else None
        low, high = (estimate-zcrit*se, estimate+zcrit*se) if se > 0 else (None, None)
        rows.append([name, estimate, se, stat, pvalue, low, high,
                     "available" if se > 0 else "unavailable: zero first-order delta variance"])
    return table(rows, columns=[label, "estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper", "inference_status"])


def _assemble(state):
    columns, options, inputs, fit, derived = (state[key] for key in ("columns", "options", "inputs", "fit", "derived"))
    n, level = len(inputs["y"]), state["level"]
    names = _parameter_names(columns, options)
    values, covariance = derived["values"], derived["covariance"]
    input_rows = [[i, inputs["y"][i], inputs["a"][i], inputs["m"][i], *inputs["c"][i]] for i in range(n)]
    means = _inference_rows(MEANS, values[:4], [row[:4] for row in covariance[:4]], level, label="mean", test_zero=False)
    means = means.drop(columns=["z", "p_value"])
    means.insert(1, "outcome_treatment", [0, 1, 0, 1])
    means.insert(2, "mediator_treatment", [0, 0, 1, 1])
    parameters = _inference_rows(names, fit["theta"], fit["covariance"], level, label="parameter", nuisance="outcome:log_sigma")
    if options["outcome_model"] == "gaussian":
        _, zcrit = _confidence(level)
        se = parameters.iloc[-1]["std_error"]
        bounds = (fit["theta"][-1]-zcrit*se, fit["theta"][-1]+zcrit*se)
        if not all(-745 < value < math.log(float(torch.finfo(DT).max)) for value in bounds):
            _error("Gaussian residual-scale confidence limits exceed representable float64 support.", "numerical_failure")
    frames = dict(
        inputs=table(input_rows, columns=[_input_position_column(columns), columns["y"], columns["treatment"], columns["mediator"], *columns["controls"]]),
        parameters=parameters, information=_matrix(fit["information"], names), bread=_matrix(fit["bread"], names),
        scores=table([[i, *row] for i, row in enumerate(fit["scores"])], columns=["row", *names]),
        model_loglikelihood=table([[i, *row] for i, row in enumerate(fit["row_loglikelihood"])], columns=["row", "mediator", "outcome"]),
        covariance=_matrix(fit["covariance"], names),
        counterfactual_rows=table([[i, *row] for i, row in enumerate(derived["counterfactual_rows"])], columns=["row", *COUNTERFACTUAL_COLUMNS]),
        means=means, means_covariance=_matrix([row[:4] for row in covariance[:4]], MEANS),
        effects=_inference_rows(EFFECTS, values[4:], [row[4:] for row in covariance[4:]], level, label="effect"),
        effects_covariance=_matrix([row[4:] for row in covariance[4:]], EFFECTS),
        delta_jacobian=table([[name, *row] for name, row in zip(MEANS+EFFECTS, derived["jacobian"])], columns=["target", *names]),
        fit_summary=table([[key, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)]
                           for key, value in sorted(fit.items()) if key not in {"theta", "covariance", "information", "bread", "scores", "row_loglikelihood"}],
                          columns=["setting", "json"]),
    )
    return TableSet(frames, title="Binary mediator standardization", method="mediation_binary", contract="binary_mediation_v1",
                    settings=state["settings"], binary_mediation_state=state, notes=NOTES)


@procedure
def mediation_binary(*, data, y, treatment, mediator, controls=None, mediator_link="logit",
                     outcome_model="gaussian", interaction=True, covariance="HC0", level=.95,
                     interpretation="associational", assumptions=None, device="cpu", weights=None,
                     cluster=None, max_iterations=200, tolerance=1e-9):
    """Fit binary-mediator standardization; causal labels require explicit assumptions."""
    if device != "cpu":
        _error("Binary mediation supports explicit native CPU float64 only.", "unsupported_device")
    if weights is not None:
        _error("Weights are outside the iid binary mediation contract.", "unsupported_weights")
    if cluster is not None:
        _error("Cluster inference is outside the iid binary mediation contract.", "unsupported_cluster")
    level, _ = _confidence(level)
    options = _options(mediator_link, outcome_model, interaction, covariance, interpretation, assumptions,
                       max_iterations, tolerance)
    columns = _column_names(y, treatment, mediator, controls)
    _parameter_names(columns, options)
    yv, mv, av, cv, plan = _inputs(data, columns, options)
    fit = fit_joint(yv, mv, av, cv, mediator_link=mediator_link, outcome_model=outcome_model,
                    interaction=interaction, covariance=covariance, max_iterations=options["max_iterations"], tolerance=options["tolerance"])
    theta = torch.tensor(fit["theta"], dtype=DT, device="cpu")
    cov = torch.tensor(fit["covariance"], dtype=DT, device="cpu")
    derived = _derived(theta, cov, yv, mv, av, cv, options)
    state = dict(schema="binary_mediation_v1", level=level, options=options, columns=columns,
                 inputs=dict(y=yv.tolist(), m=mv.tolist(), a=av.tolist(), c=cv.tolist()), fit=fit,
                 derived=derived, settings=_settings(columns, options, len(yv), plan, level))
    state["checksum"] = _checksum(state)
    return _assemble(state)


def _finite_vector(value, count):
    return isinstance(value, list) and len(value) == count and all(
        isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v)) for v in value)


def _finite_matrix(value, rows, cols):
    return isinstance(value, list) and len(value) == rows and all(_finite_vector(row, cols) for row in value)


def _matches(saved, actual, name):
    expected = torch.tensor(actual, dtype=DT, device="cpu")
    supplied = torch.tensor(saved, dtype=DT, device="cpu")
    if supplied.shape != expected.shape:
        _error(f"Saved {name} disagrees with complete input/parameter replay.", "invalid_result")
    if name in {"information", "bread", "covariance", "natural_covariance"}:
        # Dimensionless matrix agreement uses the genuine replay diagonal,
        # never a unit-independent floor or the caller's potentially inflated
        # variance. Near-zero cross terms tolerate insignificant cancellation;
        # exact structural zeros and zero-variance coordinates remain exact.
        sd = expected.diag().abs().sqrt()
        bound = 2e-9*sd[:, None]*sd[None, :]
        valid = bool(((supplied-expected).abs() <= bound).all()) and bool((supplied[expected == 0] == 0).all())
    else:
        valid = torch.allclose(supplied, expected, rtol=2e-9, atol=0)
    if not valid:
        _error(f"Saved {name} disagrees with complete input/parameter replay.", "invalid_result")


FIT_FIELDS = {
    "theta", "covariance", "information", "bread", "scores", "row_loglikelihood",
    "log_likelihood", "equation_log_likelihood", "parameter_names", "parameter_slices",
    "n", "k_controls", "mediator_link", "outcome_model", "interaction", "covariance_type",
    "control_center", "control_scale", "equations", "convergence", "solver",
    "score_convention", "information_convention", "estimated_work", "work_limit",
    "workspace_bytes", "max_iterations", "tolerance",
}


def _fit_metadata(fit, replay, options, n, k, p):
    for key in ("log_likelihood", "equation_log_likelihood", "control_center", "control_scale"):
        value = fit[key]
        valid = (isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))) if key == "log_likelihood" else _finite_vector(value, 2 if key == "equation_log_likelihood" else k)
        if not valid:
            _error(f"Saved {key} is not a complete finite value.", "invalid_result")
        _matches(value, replay[key], key)
    for key in ("parameter_names", "parameter_slices", "n", "k_controls", "mediator_link", "outcome_model", "interaction", "covariance_type"):
        if fit[key] != replay[key] or type(fit[key]) is not type(replay[key]):
            _error(f"Saved {key} disagrees with fitted equation coordinates.", "invalid_result")
    fixed = dict(solver="native centered/scaled Newton ML; closed Gaussian normalized ML",
                 score_convention="complete uncorrected per-subject conditional likelihood scores",
                 information_convention="full observed Hessian; exact factorized cross-equation zero blocks",
                 estimated_work=n*p*p*options["max_iterations"], work_limit=2_000_000_000,
                 workspace_bytes=8*(n*(8*p+16)+20*p*p),
                 max_iterations=options["max_iterations"], tolerance=options["tolerance"])
    if any(fit[key] != value or type(fit[key]) is not type(value) for key, value in fixed.items()):
        _error("Saved solver/work/options metadata is inconsistent.", "invalid_result")
    if not isinstance(fit["equations"], dict) or set(fit["equations"]) != {"mediator", "outcome"}:
        _error("Saved equation convergence metadata is incomplete.", "invalid_result")
    diagnostics = {}
    diagnostic_keys = {"iterations", "backtracks", "score_decrement", "parameter_step", "converged"}
    for j, name in enumerate(("mediator", "outcome")):
        equation = fit["equations"][name]
        if not isinstance(equation, dict) or set(equation) != diagnostic_keys | {"parameter_start", "parameter_stop", "log_likelihood"}:
            _error("Saved equation diagnostics have unsupported fields.", "invalid_result")
        if equation["converged"] is not True or type(equation["iterations"]) is not int or not 1 <= equation["iterations"] <= options["max_iterations"] or type(equation["backtracks"]) is not int or not 0 <= equation["backtracks"] <= 50*options["max_iterations"]:
            _error("Saved equation convergence counts are invalid.", "invalid_result")
        if any(isinstance(equation[key], bool) or not isinstance(equation[key], Real) or not math.isfinite(float(equation[key])) or equation[key] < -1e-15 for key in ("score_decrement", "parameter_step")):
            _error("Saved equation convergence diagnostics are not finite nonnegative values.", "invalid_result")
        if [equation["parameter_start"], equation["parameter_stop"]] != replay["parameter_slices"][name] or equation["log_likelihood"] != fit["equation_log_likelihood"][j]:
            _error("Saved equation slices/likelihood do not match complete replay.", "invalid_result")
        diagnostics[name] = {key: equation[key] for key in diagnostic_keys}
    if fit["convergence"] != dict(converged=True, **diagnostics):
        _error("Saved convergence summaries disagree with the complete equation diagnostics.", "invalid_result")


def _restore_state(state):
    required = {"schema", "level", "options", "columns", "inputs", "fit", "derived", "settings", "checksum"}
    if not isinstance(state, dict) or set(state) != required or state["schema"] != "binary_mediation_v1":
        _error("Unsupported binary mediation saved schema or fields.", "invalid_result")
    options = state["options"]
    option_keys = {"mediator_link", "outcome_model", "interaction", "covariance", "interpretation", "assumptions", "max_iterations", "tolerance"}
    if not isinstance(options, dict) or set(options) != option_keys:
        _error("Saved binary mediation options have unsupported fields.", "invalid_result")
    options = _options(**options)
    columns = state["columns"]
    if not isinstance(columns, dict) or set(columns) != {"y", "treatment", "mediator", "controls"}:
        _error("Saved binary mediation column schema is incomplete.", "invalid_result")
    columns = _column_names(**columns)
    inputs = state["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"y", "m", "a", "c"} or not isinstance(inputs["y"], list):
        _error("Saved binary mediation inputs have unsupported fields.", "invalid_result")
    n, k = len(inputs["y"]), len(columns["controls"])
    _plan(n, k, options)
    p = 2+k + 3+int(options["interaction"])+k+int(options["outcome_model"] == "gaussian")
    if not all(_finite_vector(inputs[key], n) for key in ("y", "m", "a")) or not _finite_matrix(inputs["c"], n, k):
        _error("Saved input arrays require complete finite numeric shapes.", "invalid_result")
    fit, derived = state["fit"], state["derived"]
    if not isinstance(fit, dict) or set(fit) != FIT_FIELDS:
        _error("Saved joint fit is incomplete.", "invalid_result")
    if not _finite_vector(fit["theta"], p) or not _finite_matrix(fit["scores"], n, p) or not _finite_matrix(fit["row_loglikelihood"], n, 2) or not all(
        _finite_matrix(fit[key], p, p) for key in ("information", "bread", "covariance")
    ):
        _error("Saved fit arrays require full finite original-coordinate dimensions.", "invalid_result")
    derived_keys = {"values", "jacobian", "covariance", "counterfactual_rows", "natural_parameters", "natural_covariance"}
    if not isinstance(derived, dict) or set(derived) != derived_keys or not _finite_vector(derived["values"], 9) or not _finite_matrix(derived["jacobian"], 9, p) or not _finite_matrix(derived["covariance"], 9, 9) or not _finite_matrix(derived["counterfactual_rows"], n, 12) or not _finite_vector(derived["natural_parameters"], p) or not _finite_matrix(derived["natural_covariance"], p, p):
        _error("Saved target arrays require complete finite dimensions.", "invalid_result")
    try:
        if not isinstance(state["checksum"], str) or not hmac.compare_digest(_checksum(state), state["checksum"]):
            _error("Saved binary mediation checksum does not match.", "invalid_result")
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved binary mediation state is not finite canonical JSON.") from error
    level, _ = _confidence(state["level"])
    data = {columns["y"]: inputs["y"], columns["treatment"]: inputs["a"], columns["mediator"]: inputs["m"],
            **{name: [row[j] for row in inputs["c"]] for j, name in enumerate(columns["controls"])}}
    y, m, a, c, _ = _inputs(data, columns, options)
    theta = torch.tensor(fit["theta"], dtype=DT, device="cpu")
    cov = torch.tensor(fit["covariance"], dtype=DT, device="cpu")
    for name in ("information", "bread", "covariance"):
        matrix = torch.tensor(fit[name], dtype=DT, device="cpu")
        if not torch.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-13):
            _error(f"Saved {name} is not symmetric.", "invalid_covariance")
        if float(torch.linalg.eigvalsh(matrix)[0]) < -1e-10*max(float(matrix.abs().max()), 1e-300):
            _error(f"Saved {name} is not positive semidefinite.", "invalid_covariance")
    replay = evaluate_joint(theta, y, m, a, c, mediator_link=options["mediator_link"], outcome_model=options["outcome_model"],
                            interaction=options["interaction"], covariance=options["covariance"])
    for key in ("scores", "information", "bread", "covariance", "row_loglikelihood"):
        _matches(fit[key], replay[key], key)
    _fit_metadata(fit, replay, options, n, k, p)
    from .binary_kernels import stationarity_check
    stationarity_check(theta, y, m, a, c, mediator_link=options["mediator_link"], outcome_model=options["outcome_model"],
                       interaction=options["interaction"], tolerance=options["tolerance"])
    recomputed = _derived(theta, cov, y, m, a, c, options)
    for key in derived_keys:
        _matches(derived[key], recomputed[key], key)
    if not isinstance(state["settings"], dict) or "resources" not in state["settings"]:
        _error("Saved settings are incomplete.", "invalid_result")
    resources = state["settings"]["resources"]
    if not isinstance(resources, dict) or type(resources.get("budget_bytes")) is not int or resources["budget_bytes"] < 1 or resources != _plan(n, k, options, budget_bytes=resources["budget_bytes"]):
        _error("Saved resource settings disagree with the complete buffer/work plan.", "invalid_result")
    if state["settings"] != _settings(columns, options, n, resources, level):
        _error("Saved settings disagree with the complete sample/options contract.", "invalid_result")
    return state


@procedure
def mediation_binary_restore(result, *, level=None):
    """Validate and restore all saved tables without an optimizer or refit."""
    attrs = result.attrs if isinstance(result, TableSet) else result
    if not isinstance(attrs, Mapping) or set(attrs) != {"method", "contract", "settings", "binary_mediation_state", "notes"}:
        _error("Restore requires the complete binary mediation fit attrs or TableSet.", "invalid_result")
    state = _restore_state(attrs["binary_mediation_state"])
    expected = _assemble(state)
    if dict(attrs) != expected.attrs:
        _error("Saved result attrs disagree with the sealed complete state.", "invalid_result")
    if isinstance(result, TableSet):
        if set(result) != set(expected) or any(not isinstance(result[name], pd.DataFrame) or not result[name].equals(expected[name]) for name in expected):
            _error("Saved result tables disagree with the sealed complete state.", "invalid_result")
    if level is None or level == state["level"]:
        return expected
    new_level, _ = _confidence(level)
    changed = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    changed["level"] = new_level
    changed["settings"]["level"] = new_level
    changed["checksum"] = _checksum(changed)
    return _assemble(changed)
