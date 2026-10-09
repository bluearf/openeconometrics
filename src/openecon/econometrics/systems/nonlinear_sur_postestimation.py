"""Saved nonlinear Gaussian systems: conditional targets and joint deltas.

Query rows, conditioning outcomes and standardization weights are fixed.
Mean uncertainty is parameter uncertainty; the separately reported residual
covariance describes random response variation, not a mean confidence interval.
Every target retains the complete physical mean/covariance parameter gradient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
import torch

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.econometrics.postest.index_codec import encode
from openecon.engines.distributions import chi2_sf
from openecon.resources import plan_workspace, tensor_bytes, workspace_budget_bytes

DT = torch.float64


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        _error(f"{name} must be a positive integer.", "invalid_resource_budget")
    return int(value)


def _confidence(level, state):
    if level is None:
        level = state.get("level", state.get("options", {}).get("level", 0.95))
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < level < 1:
        _error("level must lie strictly between zero and one.", "invalid_option")
    level = float(level)
    z = float(torch.special.ndtri(torch.tensor((1 + level) / 2, dtype=DT, device="cpu")))
    if not math.isfinite(z):
        _error("level is too close to one for float64 inference.", "invalid_option")
    return level, z


def _names(value, available, what, *, empty=False):
    if isinstance(value, str) and what == "x":
        value = [value]
    if value is None and empty:
        value = []
    if not isinstance(value, (list, tuple)) or (not value and not empty) \
            or any(not isinstance(name, str) or name not in available for name in value) \
            or len(set(value)) != len(value):
        _error(f"{what} must list distinct supported names: {', '.join(available)}.")
    return list(value)


def _numeric(series, name):
    if is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype) \
            or not is_numeric_dtype(series.dtype):
        _error(f"Query column '{name}' must contain real numeric values.", "non_numeric_column")
    if bool(series.isna().any()):
        _error(f"Query column '{name}' contains missing values; no query rows are dropped.",
               "missing_values")
    try:
        with torch.inference_mode(False):
            values = torch.tensor(series.to_numpy(dtype="float64", copy=True), dtype=DT, device="cpu")
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("non_numeric_column", f"Query column '{name}' is not numeric.") from exc
    if not bool(torch.isfinite(values).all()):
        _error(f"Query column '{name}' contains nonfinite values.", "non_finite_values")
    return values


def _saved(result, level):
    from .nonlinear_sur import _validated

    state, p = _validated(result, level=level)
    with torch.inference_mode(False):
        theta = torch.tensor(state["fit"]["params"], dtype=DT, device="cpu")
        covariance = torch.tensor(state["fit"]["covariance"], dtype=DT, device="cpu")
    level, z = _confidence(level, state)
    return state, p, theta, covariance, level, z


def _query_frame(p, data):
    if data is not None:
        frame = _coerce_frame(data)
    else:
        values = {name: value.detach().tolist() for name, value in p.x.items()}
        values.update({eq["y"]: p.y[:, j].detach().tolist()
                       for j, eq in enumerate(p.equations)})
        frame = pd.DataFrame(values)
        index = getattr(p, "index", None)
        if index is not None:
            frame.index = index.copy()
    if frame.columns.has_duplicates:
        _error("Query data must have distinct column names.", "duplicate_columns")
    if len(frame) == 0:
        _error("Query data must contain at least one row.", "empty_data")
    # Validate supported typed identities without stringifying them; repeated
    # identities remain distinct query positions.
    for label in frame.index:
        encode(label)
    return frame


def _geometry(p, given):
    labels = [eq["name"] for eq in p.equations]
    names = _names(given, labels, "given", empty=True)
    observed = [labels.index(name) for name in names]
    remaining = [j for j in range(p.m) if j not in observed]
    if not remaining:
        _error("Conditioning must leave at least one equation to predict.", "no_support")
    return labels, names, observed, remaining


def _plan(operation, p, n, targets, max_work, max_bytes, *, mixed=False, extra_columns=0,
          tape_operations=None):
    max_work, max_bytes = _count(max_work, "max_work"), _count(max_bytes, "max_bytes")
    k = len(p.parameter_names)
    tapes = (sum(len(eq["formula"]) for eq in p.equations) if tape_operations is None
             else tape_operations)
    work = targets * targets * max(1, k) + n * max(1, tapes) * max(1, k) * (16 if mixed else 4)
    if work > max_work:
        _error(f"Complete joint targets require an estimated {work:,} work units, exceeding "
               f"max_work={max_work:,}; reduce the query or raise the declared budget.",
               "resource_budget")
    record = plan_workspace(operation, {
        "query_numeric_inputs": tensor_bytes((n, max(1, len(p.columns)) + extra_columns)) * 3,
        "differentiable_formula_buffers": n * max(1, tapes) * max(1, k) * (64 if mixed else 32),
        "physical_parameter_covariance": tensor_bytes((k, k)) * 6,
        "complete_target_jacobian": tensor_bytes((targets, k)) * 6,
        "complete_target_covariance": tensor_bytes((targets, targets)) * 4,
    }, budget_bytes=min(max_bytes, workspace_budget_bytes())).record()
    record.update(estimated_work=work, max_work=max_work, max_bytes=max_bytes)
    return record


@torch.inference_mode(False)
def _inputs(frame, p, observed):
    required = list(dict.fromkeys([*p.columns, *(p.equations[j]["y"] for j in observed)]))
    absent = [name for name in required if name not in frame]
    if absent:
        _error(f"Required query columns are absent: {', '.join(absent)}.", "missing_columns")
    values = {name: _numeric(frame[name], name) for name in required}
    x = {name: values[name] for name in p.columns}
    if not x:
        # The safe evaluator ignores this private row marker. It supplies the
        # new query length for parameter-only/intercept-only mean systems.
        x = {"__nlsur_query_rows__": torch.zeros(len(frame), dtype=DT, device="cpu")}
    y = (torch.stack([values[p.equations[j]["y"]] for j in observed], dim=1)
         if observed else torch.empty((len(frame), 0), dtype=DT, device="cpu"))
    return x, y


def _conditional(theta, p, x, y, observed, remaining):
    from .nonlinear_sur import _means, _sigma

    means = _means(theta, p, x=x)
    sigma = _sigma(theta, p)
    if not bool(torch.isfinite(means).all() & torch.isfinite(sigma).all()):
        _error("Query means exceed the finite smooth formula domain.", "prediction_domain")
    block = sigma[remaining][:, remaining]
    answer = means[:, remaining]
    if observed:
        cross = sigma[remaining][:, observed]
        order = observed + remaining
        factor, info = torch.linalg.cholesky_ex(sigma[order][:, order])
        if int(info) or not bool(torch.isfinite(factor).all()):
            _error("The conditioning covariance is not positive definite.", "invalid_result")
        no = len(observed)
        loading = torch.cholesky_solve(cross.T, factor[:no, :no]).T
        answer = answer + (y - means[:, observed]) @ loading.T
        # The trailing Cholesky block is the conditional residual factor.
        # Its product avoids subtracting nearly equal covariance matrices.
        residual_factor = factor[no:, no:]
        block = residual_factor @ residual_factor.T
    block = (block + block.T) / 2
    if not bool(torch.isfinite(answer).all() & torch.isfinite(block).all()) \
            or bool((block.diagonal() <= 0).any()):
        _error("Conditional means or residual covariance exceed finite positive support.",
               "prediction_domain")
    return answer, block


def _jacobian(function, theta):
    with torch.inference_mode(False), torch.enable_grad():
        variable = theta.detach().clone().requires_grad_(True)
        value = function(variable)
        columns = []
        # Directional columns keep the complete row target without an N-by-N
        # reverse Jacobian; jvp uses Torch double-backward on this small input.
        for j in range(len(theta)):
            direction = torch.zeros_like(variable)
            direction[j] = 1
            _, column = torch.autograd.functional.jvp(function, variable, direction,
                                                       create_graph=False, strict=False)
            columns.append(column)
        jacobian = torch.stack(columns, dim=1)
    value, jacobian = value.detach(), jacobian.detach()
    if not bool(torch.isfinite(value).all() & torch.isfinite(jacobian).all()):
        _error("Query values or derivatives leave the finite smooth formula domain.",
               "prediction_domain")
    return value, jacobian


def _delta(jacobian, covariance):
    # Unit-scaled PSD factor retains small variance coordinates and exact
    # fixed/constraint directions. It does not ridge a singular covariance.
    covariance = (covariance + covariance.T) / 2
    diagonal = covariance.diag()
    if not bool(torch.isfinite(covariance).all()) or bool((diagonal < 0).any()):
        _error("Saved covariance is not finite positive semidefinite.", "invalid_result")
    positive = diagonal > 0
    if bool((~positive).any()) and bool((covariance[~positive] != 0).any()):
        _error("A fixed parameter has nonzero saved cross covariance.", "invalid_result")
    scale = torch.where(positive, diagonal.sqrt(), torch.ones_like(diagonal))
    correlation = covariance / scale[:, None] / scale[None, :]
    eigenvalues, vectors = torch.linalg.eigh(correlation)
    if float(eigenvalues.min()) < -1e-10 * max(1.0, float(eigenvalues.abs().max())):
        _error("Saved covariance is not positive semidefinite.", "invalid_result")
    factor = scale[:, None] * vectors * eigenvalues.clamp_min(0).sqrt()[None, :]
    mapped = jacobian @ factor
    output = mapped @ mapped.T
    if not bool(torch.isfinite(output).all()):
        _error("Joint delta covariance exceeds finite float64 support.", "numerical_failure")
    return (output + output.T) / 2


def _matrix(value, rows, columns=None):
    columns = rows if columns is None else columns
    marker = "target"
    while marker in columns:
        marker = "_" + marker
    return table([[name, *row] for name, row in zip(rows, value.tolist(), strict=True)],
                 columns=[marker, *columns])


def _interval(value, variance, z, *, positive=False):
    se = math.sqrt(max(0.0, float(variance)))
    if se == 0:
        return dict(std_error=se, ci_low=None, ci_high=None,
                    inference="unavailable: zero first-order delta variance")
    if positive:
        radius = z * se / value
        lowlog, highlog = math.log(value) - radius, math.log(value) + radius
        try:
            lower, upper = math.exp(lowlog), math.exp(highlog)
        except OverflowError:
            _error("Log-variance confidence limits exceed finite float64 support.",
                   "numerical_failure")
        kind = "asymptotic normal on log variance"
    else:
        lower, upper = value - z * se, value + z * se
        kind = "asymptotic normal"
    if not all(math.isfinite(v) for v in (se, lower, upper)):
        _error("Confidence limits exceed finite float64 support.", "numerical_failure")
    answer = dict(std_error=se, ci_low=lower, ci_high=upper, inference=kind)
    if positive:
        answer.update(log_ci_low=lowlog, log_ci_high=highlog)
    return answer


def _output(method, frames, state, p, frame, given, level, resources, **extra):
    return TableSet(frames, title="Nonlinear SUR saved query", method=method,
                    contract="nonlinear_sur_postestimation_v1",
                    settings=dict(level=level, given=given, precision="float64", device="cpu",
                                  physical_parameters=p.parameter_names,
                                  source_fit_checksum=state.get("checksum"),
                                  row_index_codes=[encode(label) for label in frame.index],
                                  row_index_names=list(frame.index.names),
                                  missing="raise", query_rows=len(frame), refit=False,
                                  no_optimizer=True, inference_df=None,
                                  uncertainty="full joint physical mean and residual covariance parameter delta",
                                  query_inputs_and_weights="fixed, not estimated",
                                  resources=resources, **extra),
                    notes=["Mean confidence intervals describe parameter uncertainty.",
                           "Residual covariance describes random response variation, not mean uncertainty.",
                           "Conditional outcomes and all query covariates are held fixed."])


@procedure
def nlsur_predict(result, data=None, *, given=None, level=None,
                  max_work=100000000, max_bytes=256000000):
    """Saved equation means and residual covariance, optionally Gaussian conditional.

    ``given`` lists equation labels whose outcome columns are observed in each
    query row. Every remaining equation is predicted. The complete mean and
    unique residual-covariance target vector has one joint delta covariance.
    """
    state, p, theta, covariance, level, z = _saved(result, level)
    frame = _query_frame(p, data)
    labels, given, observed, remaining = _geometry(p, given)
    pairs = [(i, j) for i in range(len(remaining)) for j in range(i + 1)]
    count = len(frame) * len(remaining)
    resources = _plan("nlsur_predict", p, len(frame), count + len(pairs), max_work,
                      max_bytes, extra_columns=len(observed))
    x, y = _inputs(frame, p, observed)

    def targets(t):
        means, sigma = _conditional(t, p, x, y, observed, remaining)
        return torch.cat([means.flatten(), torch.stack([sigma[i, j] for i, j in pairs])])

    values, jacobian = _jacobian(targets, theta)
    target_covariance = _delta(jacobian, covariance)
    names = [f"mean_{i + 1}" for i in range(count)] + [f"residual_{i + 1}" for i in range(len(pairs))]
    rows = []
    for row in range(len(frame)):
        for position, equation in enumerate(remaining):
            i = row * len(remaining) + position
            rows.append(dict(target=names[i], row_position=row, equation=labels[equation],
                             estimate=float(values[i]),
                             **_interval(float(values[i]), target_covariance[i, i], z)))
    predictions = table(rows)
    predictions.index = frame.index.repeat(len(remaining))
    residuals = []
    for k, (i, j) in enumerate(pairs):
        position = count + k
        value = float(values[position])
        residuals.append(dict(target=names[position], equation=labels[remaining[i]],
                              other_equation=labels[remaining[j]], estimate=value,
                              **_interval(value, target_covariance[position, position], z,
                                          positive=(i == j))))
    _, residual = _conditional(theta, p, x, y, observed, remaining)
    frames = dict(means=predictions, residual_covariance=table(residuals),
                  parameter_jacobian=_matrix(jacobian, names, p.parameter_names),
                  target_covariance=_matrix(target_covariance, names),
                  residual_variation=_matrix(residual, [labels[i] for i in remaining]))
    return _output("nlsur_predict", frames, state, p, frame, given, level, resources,
                   target_order=names, remaining=[labels[i] for i in remaining],
                   target_layout="row-major remaining means, then lower-triangle residual covariance",
                   mean_interval="parameter delta only; excludes random residual variation")


@torch.inference_mode(False)
def _weights(frame, weights):
    if weights is None:
        values = torch.ones(len(frame), dtype=DT, device="cpu")
        source = "equal fixed weights"
    elif isinstance(weights, str):
        if weights not in frame:
            _error("The fixed standardization weight column is absent.", "missing_columns")
        values = _numeric(frame[weights], weights)
        source = f"fixed query column: {weights}"
    else:
        if isinstance(weights, pd.Series):
            if not weights.index.equals(frame.index):
                _error("Fixed weight Series index must equal the complete query index in order.",
                       "invalid_weights")
            series = weights
        elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes, Mapping)):
            series = pd.Series(list(weights))
        elif hasattr(weights, "ndim") and weights.ndim == 1:
            series = pd.Series(weights.tolist())
        else:
            _error("weights must be a numeric column or a complete positional vector.",
                   "invalid_weights")
        if len(series) != len(frame):
            _error("Fixed weights must cover every query position.", "invalid_weights")
        values = _numeric(series, "weights")
        source = "fixed positional weights"
    if bool((values < 0).any()) or not bool((values > 0).any()):
        _error("Fixed weights must be nonnegative with positive total mass.", "invalid_weights")
    # Normalize in two steps to avoid overflow under harmless common rescaling.
    normalized = values / values.max()
    normalized = normalized / normalized.sum()
    if not bool(torch.isfinite(normalized).all()):
        _error("Fixed weights exceed finite float64 support.", "invalid_weights")
    return values, normalized, source


def _effects(theta, p, x, y, observed, remaining, variables, scale):
    with torch.enable_grad():
        changed = {name: values.detach().clone().requires_grad_(True) for name, values in x.items()}
        means, _ = _conditional(theta, p, changed, y, observed, remaining)
        blocks = []
        for equation in range(len(remaining)):
            derivatives = []
            for variable in variables:
                gradient, = torch.autograd.grad(means[:, equation].sum(), changed[variable],
                                                create_graph=True, retain_graph=True,
                                                allow_unused=True)
                if gradient is None:
                    gradient = theta.sum() * 0 + torch.zeros_like(changed[variable])
                if scale == "elasticity":
                    if bool((changed[variable] <= 0).any()) or bool((means[:, equation] <= 0).any()):
                        _error("Elasticities require strictly positive x and predicted mean in every "
                               "requested row/equation; no support denominator is truncated.",
                               "no_support")
                    gradient = gradient * changed[variable] / means[:, equation]
                derivatives.append(gradient)
            blocks.append(torch.stack(derivatives, dim=1))
        return torch.stack(blocks, dim=1)


@procedure
def nlsur_margins(result, data=None, *, x, scale="effect", given=None, weights=None, level=None,
                  max_work=100000000, max_bytes=256000000):
    """Continuous conditional/unconditional effects and fixed-weight averages.

    ``x`` is a feature name or a distinct list. ``scale='elasticity'`` is
    x/mean times the derivative and requires both values positive everywhere.
    Nonnegative weights cover all query positions; zero weights remain in the
    per-case output and never remove a domain check.
    """
    if scale not in ("effect", "elasticity"):
        _error("scale must be effect or elasticity.", "invalid_option")
    state, p, theta, covariance, level, z = _saved(result, level)
    frame = _query_frame(p, data)
    labels, given, observed, remaining = _geometry(p, given)
    variables = _names(x, p.columns, "x")
    width = len(remaining) * len(variables)
    count = len(frame) * width
    resources = _plan("nlsur_margins", p, len(frame), count + width, max_work, max_bytes,
                      mixed=True, extra_columns=len(observed) + int(weights is not None))
    columns, y = _inputs(frame, p, observed)
    raw_weights, normalized, weight_source = _weights(frame, weights)

    def targets(t):
        effect = _effects(t, p, columns, y, observed, remaining, variables, scale)
        average = (effect * normalized[:, None, None]).sum(0)
        return torch.cat([effect.flatten(), average.flatten()])

    values, jacobian = _jacobian(targets, theta)
    target_covariance = _delta(jacobian, covariance)
    names = [f"effect_{i + 1}" for i in range(count)] + [f"average_{i + 1}" for i in range(width)]
    rows, averages = [], []
    for row in range(len(frame)):
        for e, equation in enumerate(remaining):
            for v, variable in enumerate(variables):
                i = row * width + e * len(variables) + v
                rows.append(dict(target=names[i], row_position=row, equation=labels[equation],
                                 x=variable, scale=scale, weight=float(raw_weights[row]),
                                 normalized_weight=float(normalized[row]), estimate=float(values[i]),
                                 **_interval(float(values[i]), target_covariance[i, i], z)))
    for e, equation in enumerate(remaining):
        for v, variable in enumerate(variables):
            i = count + e * len(variables) + v
            averages.append(dict(target=names[i], equation=labels[equation], x=variable,
                                 scale=scale, estimate=float(values[i]), query_rows=len(frame),
                                 positive_weight_rows=int((raw_weights > 0).sum()),
                                 normalized_weight_sum=float(normalized.sum()),
                                 **_interval(float(values[i]), target_covariance[i, i], z)))
    effects = table(rows)
    effects.index = frame.index.repeat(width)
    frames = {"effects" if scale == "effect" else "elasticities": effects,
              "averages": table(averages),
              "parameter_jacobian": _matrix(jacobian, names, p.parameter_names),
              "target_covariance": _matrix(target_covariance, names)}
    return _output("nlsur_margins", frames, state, p, frame, given, level, resources,
                   target_order=names, x=variables, scale=scale, weight_source=weight_source,
                   fixed_normalized_weights=normalized.tolist(),
                   support="every query row and remaining equation; no implicit exclusions",
                   average_definition="fixed-weight average of per-case effects, then joint parameter delta",
                   target_layout="row/equation/feature effects, then equation/feature averages")


def _expressions(expressions, p):
    from .nonlinear_sur import _parse_smooth

    if not isinstance(expressions, Mapping) or not expressions or len(expressions) > 256 \
            or any(not isinstance(name, str) or not name.strip() for name in expressions):
        _error("expressions must map one through 256 distinct target labels to safe formulas.",
               "invalid_expression")
    parsed = []
    for label, text in expressions.items():
        formula = _parse_smooth(text)
        if formula.columns or set(formula.parameters) - set(p.parameter_names) or formula.start:
            _error("Contrast expressions use fitted parameters in braces only; no data columns "
                   "or starting-value assignments.", "invalid_expression")
        parsed.append((label, formula))
    return parsed


def _null(null, labels):
    if null is None:
        values = [0.0] * len(labels)
    elif isinstance(null, Mapping):
        if set(null) != set(labels):
            _error("null must supply exactly one value for every expression.", "invalid_restrictions")
        values = [null[label] for label in labels]
    elif isinstance(null, (list, tuple)) and len(null) == len(labels):
        values = list(null)
    elif len(labels) == 1 and isinstance(null, Real):
        values = [null]
    else:
        _error("null must be a complete target mapping or positional vector.", "invalid_restrictions")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(float(value)) for value in values):
        _error("Every null value must be a finite real number.", "invalid_restrictions")
    return torch.tensor(values, dtype=DT, device="cpu")


def _contrast_values(theta, p, parsed):
    from .nonlinear_sur import _evaluate

    values = []
    for _, formula in parsed:
        selected = torch.stack([theta[p.parameter_names.index(name)] for name in formula.parameters])
        value = _evaluate(formula, {}, selected, 1)
        values.append(value[0])
    return torch.stack(values)


@procedure
def nlsur_contrast(result, expressions, *, null=None, level=None,
                   max_work=100000000, max_bytes=256000000):
    """Joint local first-order Wald inference for smooth safe parameter expressions.

    Parameter names, including physical covariance aliases, appear in braces.
    Hypotheses must be regular with positive-definite target covariance. A
    zero residual-variance boundary is not an ordinary chi-square hypothesis.
    """
    state, p, theta, covariance, level, z = _saved(result, level)
    parsed = _expressions(expressions, p)
    labels = [label for label, _ in parsed]
    nulls = _null(null, labels)
    pairs = [(i, j) for i in range(p.m) for j in range(i + 1)]
    diagonal = {p.parameter_names[p.q + k] for k, (i, j) in enumerate(pairs) if i == j}
    for k, (_, formula) in enumerate(parsed):
        if len(formula.tape) == 1 and formula.tape[0][0] == "par" \
                and formula.parameters[0] in diagonal and float(nulls[k]) <= 0:
            _error("Direct residual-variance hypotheses must have a positive interior null; "
                   "zero is a nonregular covariance boundary.", "nonregular_hypothesis")
    # Expressions are bounded by the same parser limit as fitting. Their live
    # differentiable tape replaces the fitted-row tape estimate here.
    operations = sum(len(formula.tape) for _, formula in parsed)
    resources = _plan("nlsur_contrast", p, 1, len(parsed), max_work, max_bytes,
                      mixed=True, tape_operations=operations)
    resources["expression_tape_operations"] = operations
    values, jacobian = _jacobian(lambda t: _contrast_values(t, p, parsed), theta)
    target_covariance = _delta(jacobian, covariance)
    diagonal_variance = target_covariance.diag()
    if bool((diagonal_variance <= 0).any()):
        _error("Wald expressions have zero first-order variance; choose estimable regular "
               "hypotheses.", "nonestimable_restriction")
    scale = diagonal_variance.sqrt()
    correlation = target_covariance / scale[:, None] / scale[None, :]
    eigenvalues = torch.linalg.eigvalsh(correlation)
    if float(eigenvalues.min()) <= 64 * len(labels) * torch.finfo(DT).eps \
            * max(1.0, float(eigenvalues.max())):
        _error("Joint Wald expressions are rank deficient; remove dependent restrictions.",
               "nonestimable_restriction")
    factor = torch.linalg.cholesky(correlation)
    difference = (values - nulls) / scale
    solved = torch.linalg.solve_triangular(factor, difference[:, None], upper=False)[:, 0]
    statistic = float(solved @ solved)
    if not math.isfinite(statistic):
        _error("Wald statistic exceeds finite float64 support.", "numerical_failure")
    rows = []
    for i, label in enumerate(labels):
        individual = float(difference[i])
        rows.append(dict(target=label, expression=parsed[i][1].source, estimate=float(values[i]),
                         null=float(nulls[i]), statistic=individual,
                         p_value=math.erfc(abs(individual) / math.sqrt(2)),
                         **_interval(float(values[i]), target_covariance[i, i], z)))
    frames = dict(contrasts=table(rows), parameter_jacobian=_matrix(jacobian, labels, p.parameter_names),
                  target_covariance=_matrix(target_covariance, labels),
                  wald=table([dict(statistic=statistic, df=len(labels), distribution="chi2",
                                   p_value=chi2_sf(statistic, len(labels)))]))
    frame = pd.DataFrame(index=pd.RangeIndex(1))
    return _output("nlsur_contrast", frames, state, p, frame, [], level, resources,
                   target_order=labels, hypothesis="declared local regular first-order restrictions",
                   null_feasibility="not established by the delta method",
                   boundary_policy="direct residual variance nulls must be positive")
