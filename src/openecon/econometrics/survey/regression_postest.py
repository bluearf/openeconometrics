"""Saved single-stage regression predictions and conditional coefficient inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import f_sf
from openecon.resources import plan_workspace, workspace_budget_bytes
from .common import FLOAT, finite, number
from .regression_common import MAX_WORK, cpu_call, inference_frame, restore


def _data(state, data, *, missing, weights=None, at=None, max_rows=None):
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_survey_option", "missing must be raise/drop.")
    if weights is not None and (not isinstance(weights, str) or not weights.strip()):
        raise AnalysisError("invalid_survey_option", "weights must name fixed evaluation weights.")
    if at is not None and (not isinstance(at, Mapping) or any(v not in state.regressors for v in at)):
        raise AnalysisError("invalid_survey_option", "at maps fitted numeric regressor names to finite scalars.")
    at = {} if at is None else {k: number(v, "at value") for k, v in at.items()}
    roles = [v for v in state.regressors if v not in at] + ([weights] if weights else [])
    limit = state.design.max_rows if max_rows is None else min(state.design.max_rows, max_rows)
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        if not roles:
            raise AnalysisError("invalid_survey_input", "Use a DataFrame with an explicit row index when every covariate is fixed.")
        if any(v not in data for v in roles):
            raise AnalysisError("missing_column", "Every evaluation role must exist.")
        if any(type(data[v]).__module__.startswith("torch") or not hasattr(data[v], "__len__") for v in roles):
            raise AnalysisError("unsupported_survey_input", "Provide finite resident columns with declared lengths.")
        if any(len(data[v]) > limit for v in roles):
            raise AnalysisError("survey_regression_budget", "Evaluation rows exceed the explicit target budget.")
        try:
            frame = pd.DataFrame({v: data[v] for v in roles})
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_data", "Evaluation columns need consistent row lengths.") from exc
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > limit:
            raise AnalysisError("survey_regression_budget", "Evaluation rows exceed the explicit target budget.")
        if any(not isinstance(row, Mapping) for row in data):
            raise AnalysisError("unsupported_survey_input", "Evaluation row records must be mappings.")
        frame = pd.DataFrame([{v: row.get(v) for v in roles} for row in data], index=range(len(data)))
    else:
        raise AnalysisError("unsupported_survey_input", "Use a resident DataFrame or column/row mappings; no Dataset/device route.")
    if frame.columns.has_duplicates or any(v not in frame for v in roles):
        raise AnalysisError("invalid_survey_columns", "Evaluation roles need unique existing columns.")
    if not len(frame):
        raise AnalysisError("empty_data", "Evaluation needs at least one row.")
    if len(frame) > limit:
        raise AnalysisError("survey_regression_budget", "Evaluation rows exceed the explicit target budget.")
    k = len(state.labels)
    if len(frame)*k*k > MAX_WORK:
        raise AnalysisError("survey_regression_budget", "Evaluation N*K*K exceeds its work budget.")
    workspace = plan_workspace("conditional survey coefficient targets",
                               {"covariates/Jacobians/serialization": len(frame)*(k+3)*192,
                                "joint target covariance": (len(frame)**2*96 if max_rows else k*k*96)},
                               budget_bytes=workspace_budget_bytes())
    planned = sum(int(frame[v].memory_usage(index=False, deep=True)) for v in roles)
    if planned > state.design.max_memory_mb*1024**2:
        raise AnalysisError("survey_regression_budget", "Evaluation roles exceed the declared resident memory budget.")
    positions, rows, evaluation_weights, exclusions = [], [], [], []
    for position, raw in enumerate(frame[roles].itertuples(index=False, name=None)):
        if any(not pd.api.types.is_scalar(v) for v in raw):
            raise AnalysisError("invalid_survey_target", "Evaluation values must be scalar.")
        if any(pd.isna(v) for v in raw):
            if missing == "raise":
                raise AnalysisError("survey_missing", "Missing evaluation covariates require missing='drop'.")
            exclusions.append(position)
            continue
        values = dict(zip(roles, raw))
        row = ([1.] if state.intercept else []) + [at[v] if v in at else number(values[v], v) for v in state.regressors]
        w = number(values[weights], "evaluation weight", 0.) if weights else 1.
        if w <= 0:
            raise AnalysisError("invalid_survey_weights", "Fixed evaluation weights must be strictly positive.")
        positions.append(position)
        rows.append(row)
        evaluation_weights.append(w)
    if not rows:
        raise AnalysisError("empty_data", "No complete evaluation rows remain.")
    X = torch.tensor(rows, dtype=FLOAT)
    w = torch.tensor(evaluation_weights, dtype=FLOAT)
    relative = w/w.max()
    w = relative/relative.sum()
    if bool((w <= 0).any()):
        raise AnalysisError("survey_numerical_failure", "Evaluation weight ratios underflowed.")
    metadata = {"physical_positions": positions, "covariate_exclusions": exclusions,
                "n_evaluation": len(rows), "missing": missing, "at": at,
                "evaluation_weight_column": weights, "evaluation_weights": w.tolist(),
                "workspace": workspace.record(), "conditional_fixed_covariates": True,
                "population_distribution_uncertainty": False}
    return frame.index.take(positions), X, w, metadata


def _link(family, index):
    if family == "linear":
        return index, torch.ones_like(index), torch.zeros_like(index)
    if family == "logit":
        mean = torch.sigmoid(index)
        first = torch.sigmoid(index)*torch.sigmoid(-index)
        return mean, first, first*(1-2*mean)
    if family == "probit":
        mean = torch.special.ndtr(index)
        first = torch.exp(-index.square()/2)/math.sqrt(2*math.pi)
        return mean, first, -index*first
    mean = finite(torch.exp(index), "Poisson conditional mean")
    return mean, mean, mean


def _target(state, estimates, jacobian, labels, *, alpha, null=0., target, metadata=None):
    finite(estimates, "Conditional survey target")
    finite(jacobian, "Conditional survey Jacobian")
    covariance = finite(jacobian @ torch.tensor(state.covariance, dtype=FLOAT) @ jacobian.T,
                        "Conditional survey covariance")
    covariance = (covariance+covariance.T)/2
    frame = inference_frame(state, estimates.tolist(), covariance.tolist(), labels,
                            alpha=alpha, null=null, target=target)
    frame.attrs.update(jacobian=jacobian.tolist(), **(metadata or {}))
    return frame


@cpu_call
def survey_predict(result, data, *, kind="response", missing="raise", alpha=None):
    """Conditional fitted means on up to 256 new rows with full coefficient-delta covariance.

    These are mean intervals, not future-outcome prediction intervals. Missing
    exclusions retain physical positions; original fit observations are unnecessary.
    """
    state = restore(result)
    if not isinstance(kind, str) or kind not in {"response", "linear"}:
        raise AnalysisError("invalid_survey_option", "kind must be response/linear.")
    labels, X, _, metadata = _data(state, data, missing=missing, max_rows=256)
    index = finite(X @ torch.tensor(state.coefficients, dtype=FLOAT), "Conditional linear prediction")
    if kind == "linear":
        estimates, jacobian = index, X
    else:
        estimates, first, _ = _link(state.family, index)
        jacobian = first[:, None]*X
    frame = _target(state, estimates, jacobian, labels, alpha=alpha,
                    target="conditional "+kind+" prediction", metadata=metadata)
    frame.attrs["kind"] = kind
    return frame


@cpu_call
def survey_margins(result, data, *, variables=None, at=None, weights=None, missing="raise", alpha=None):
    """Fixed-covariate average mean or continuous AMEs with complete joint delta covariance.

    Evaluation weights are fixed standardization weights. Unconditional sampled
    covariate uncertainty and categorical/discrete contrasts are separate targets.
    """
    state = restore(result)
    if variables is not None:
        variables = [variables] if isinstance(variables, str) else variables
        if (not isinstance(variables, (list, tuple)) or not variables
                or any(not isinstance(v, str) or v not in state.regressors for v in variables)
                or len(set(variables)) != len(variables)):
            raise AnalysisError("invalid_survey_option", "variables names distinct fitted continuous regressors.")
    _, X, w, metadata = _data(state, data, missing=missing, weights=weights, at=at)
    beta = torch.tensor(state.coefficients, dtype=FLOAT)
    mean, first, second = _link(state.family, finite(X@beta, "Margins linear predictor"))
    if variables is None:
        estimates = (w@mean).reshape(1)
        jacobian = (w@(first[:, None]*X)).reshape(1, -1)
        labels = ["average predicted response"]
    else:
        average_first = w@first
        derivative = w@(second[:, None]*X)
        estimates, gradients = [], []
        for name in variables:
            j = state.labels.index(name)
            estimates.append(beta[j]*average_first)
            gradient = beta[j]*derivative.clone()
            gradient[j] += average_first
            gradients.append(gradient)
        estimates, jacobian = torch.stack(estimates), torch.stack(gradients)
        labels = ["AME:"+v for v in variables]
    metadata.update(variables=None if variables is None else list(variables),
                    estimand="conditional fixed-covariate standardization; coefficient uncertainty only")
    return _target(state, estimates, jacobian, labels, alpha=alpha,
                   target="conditional continuous AME" if variables else "conditional average mean", metadata=metadata)


@cpu_call
def survey_lincom(result, coefficients, *, null=0., alpha=None):
    """One saved coefficient contrast with full design covariance and design-t inference."""
    state = restore(result)
    if isinstance(coefficients, Mapping):
        if any(v not in state.labels for v in coefficients):
            raise AnalysisError("invalid_survey_contrast", "Contrast names must match saved coefficient labels.")
        coefficients = [coefficients.get(v, 0.) for v in state.labels]
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(state.labels):
        raise AnalysisError("invalid_survey_contrast", "Declare one coefficient per saved parameter.")
    jacobian = torch.tensor([[number(v, "contrast coefficient") for v in coefficients]], dtype=FLOAT)
    estimate = jacobian@torch.tensor(state.coefficients, dtype=FLOAT)
    return _target(state, estimate, jacobian, ["linear contrast"], alpha=alpha,
                   null=number(null, "null"), target="linear contrast")


@cpu_call
def survey_test(result, restrictions, *, null=None):
    """Full-rank survey-adjusted joint Wald F, with denominator df=design_df-q+1.

    Singular requested covariance or q>design_df is refused; no pseudoinverse
    changes the requested null. A single restriction gives the squared design t.
    """
    state = restore(result)
    if (not isinstance(restrictions, (list, tuple)) or not restrictions
            or len(restrictions) > len(state.labels)
            or any(not isinstance(r, (list, tuple)) or len(r) != len(state.labels) for r in restrictions)):
        raise AnalysisError("invalid_survey_test", "Declare a nonempty q-by-K restriction matrix.")
    q = len(restrictions)
    if q > state.df:
        raise AnalysisError("unsupported_survey_test", "Survey-adjusted joint F requires q <= complete design df.")
    R = torch.tensor([[number(v, "restriction") for v in row] for row in restrictions], dtype=FLOAT)
    values = [0.]*q if null is None else null
    if not isinstance(values, (list, tuple)) or len(values) != q:
        raise AnalysisError("invalid_survey_test", "Declare one finite null per restriction.")
    c = torch.tensor([number(v, "null") for v in values], dtype=FLOAT)
    row_scale = R.abs().amax(1)
    if bool((row_scale == 0).any()):
        raise AnalysisError("survey_test_rank", "Restrictions must be nonzero and independent.")
    R, c = R/row_scale[:, None], c/row_scale
    V = finite(R@torch.tensor(state.covariance, dtype=FLOAT)@R.T, "Restriction covariance")
    scale = V.diag().sqrt()
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise AnalysisError("survey_test_rank", "Requested restriction covariance is singular.")
    correlation = V/scale[:, None]/scale[None, :]
    if float(torch.linalg.eigvalsh((correlation+correlation.T)/2).min()) <= 1e-12:
        raise AnalysisError("survey_test_rank", "Requested restriction covariance lacks full rank.")
    difference = finite((R@torch.tensor(state.coefficients, dtype=FLOAT)-c)/scale, "Restriction difference")
    W = number(float(difference@torch.linalg.solve(correlation, difference)), "Wald statistic", 0.)
    denominator_df = state.df-q+1
    statistic = (denominator_df/state.df)*(W/q)
    frame = pd.DataFrame([[statistic, f_sf(statistic, q, denominator_df), q, denominator_df, W]],
                         index=["joint Wald F"], columns=["statistic", "p_value", "df_num", "df_den", "wald_chi2"])
    frame.attrs.update(survey_regression_state=state.model_dump(mode="json"),
                       method="survey-adjusted Wald F", design_df=state.df,
                       formula="F=(design_df-q+1)*W/(q*design_df); reference F(q,design_df-q+1)",
                       restrictions=restrictions, null=list(values), covariance_matrix=V.tolist(),
                       restriction_row_scales=row_scale.tolist(),
                       covariance_convention="row-normalized restrictions; equivalent requested null and Wald test")
    return frame
