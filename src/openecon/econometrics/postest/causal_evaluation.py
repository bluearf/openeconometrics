"""Identified population effects and explicit nuisance/GMM outcome functions.

No unit mean is inferred from an ATT, cutoff jump or moment parameter. Supplied
populations for nuisance standardization are fixed evaluation rows; their delta
uncertainty is distinct from the original estimator's full influence covariance.
"""

from copy import deepcopy
from dataclasses import replace
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle
from openecon.resources import plan_workspace
from openecon.engines.inference import critical_value
from .advanced_prediction import Response, Phi
from .inference import _parameters, _alpha
from .linear_prediction import _coding
from .streaming_prediction import materialize_predictions


def fail(code, message):
    raise AnalysisError(code, message)


class Outcome(Response):
    def encode(self, frame, design):
        design = super().encode(frame, design)
        treatment_equations = [name for name in self.equations if name.startswith("TME")]
        if treatment_equations:
            indexes = torch.stack(
                [self.index(design, self.saved, name) for name in treatment_equations], 1
            )
            if len(treatment_equations) == 1 and self.extra.get("treatment_model") == "probit":
                p = Phi(indexes[:, 0])
                probabilities = torch.stack((1 - p, p), 1)
            else:
                probabilities = torch.cat(
                    (torch.zeros((len(indexes), 1), dtype=torch.float64), indexes), 1
                ).softmax(1)
            tolerance = self.extra.get("overlap", {}).get("pstolerance", self.tolerance)
            if not bool(torch.isfinite(probabilities).all()) or bool(
                (probabilities < tolerance).any()
            ):
                fail(
                    "unsupported_causal_support",
                    "Evaluation covariates violate the fitted treatment overlap tolerance.",
                )
        return design

    def definition(self, kind):
        return (
            "saved conditional outcome-model mean; explicit treatment="
            + self.target
            + "; full joint nuisance delta covariance; supplied evaluation population fixed"
        )

    def _values(self, design, beta, kind):
        eta = self.index(design, beta, "outcome")
        if kind in {"xb", "stdp"}:
            return eta
        model = self.extra["outcome_model"]
        if model == "linear":
            return eta
        if model == "logit":
            return eta.sigmoid()
        if model == "probit":
            return Phi(eta)
        if model == "poisson":
            return eta.exp()
        fail("prediction_state_missing", "Saved outcome model link is unavailable.")


def _nuisance_model(result, treatment):
    from .prediction import _Model

    record = result.extra.get("evaluation_state")
    if not isinstance(record, dict) or record.get("version") != 1:
        fail(
            "prediction_state_missing",
            "This saved treatment-effects result lacks full joint nuisance state; no refit is performed.",
        )
    levels = result.extra.get("levels")
    if treatment not in levels:
        fail(
            "unknown_treatment",
            "Select one explicit saved treatment label: " + ", ".join(levels) + ".",
        )
    terms, params, cov = record.get("terms"), record.get("parameters"), record.get("covariance")
    if (
        not isinstance(terms, list)
        or not isinstance(params, list)
        or len(params) != len(terms)
        or len(terms) > 384
    ):
        fail("invalid_result", "Saved joint nuisance state is malformed.")
    plan_workspace(
        "saved joint causal covariance", {"validation_and_snapshot": 96 * len(terms) ** 2}
    )
    template = result.coefficients[0]
    result.coefficients = [
        template.model_copy(update={"term": term, "equation": None, "estimate": value})
        for term, value in zip(terms, params, strict=True)
    ]
    result.covariance_matrix = cov
    state = _parameters(result)
    equation = record.get("equations", {}).get("OME" + treatment)
    if not isinstance(equation, dict) or not equation:
        fail(
            "prediction_state_missing",
            "The saved estimator has no outcome regression for this treatment; IPW/matching do not identify a unit response from their effect alone.",
        )
    treatment_x = result.spec.columns.get("tx") or []
    treatment_x = [treatment_x] if isinstance(treatment_x, str) else list(treatment_x)
    predictors = list(dict.fromkeys([*result.spec.predictors, *treatment_x]))
    features, categories = _coding(result, predictors)
    if set(equation) - set(terms) or set(equation.values()) - set(features):
        fail(
            "prediction_state_missing", "Outcome nuisance feature/parameter mapping is incomplete."
        )
    raw = {key: value for key, value in features.items() if value[0] != "constant"}
    equations = {
        "outcome": equation,
        **{name: values for name, values in record["equations"].items() if name.startswith("TME")},
    }
    if any(
        set(values.values()) - set(features) or set(values) - set(terms)
        for values in equations.values()
    ):
        fail("prediction_state_missing", "Saved treatment/outcome nuisance coding is incomplete.")
    adapter = Outcome(
        "teffects",
        state.terms,
        raw,
        equations,
        {},
        result.extra,
        treatment,
        kinds=frozenset({"xb", "stdp", "response"}),
    )
    adapter.saved = state.beta
    adapter.tolerance = result.spec.options.get("pstolerance", 1e-5)
    return _Model(
        result,
        state,
        None,
        "explicit",
        predictors,
        categories,
        {},
        {},
        set(),
        None,
        None,
        None,
        None,
        None,
        None,
        adapter,
    )


def _gmm_model(result, function):
    from openecon.econometrics.quantile.formula import parse
    from .prediction import _Model

    if not isinstance(function, str) or not function.strip():
        fail(
            "outcome_function_required",
            "General GMM needs an explicit outcome_function in the safe named-parameter formula grammar; moment residuals are not outcomes.",
        )
    formula = parse(function)
    state = _parameters(result)
    if not formula.parameters or set(formula.parameters) - set(state.terms):
        fail(
            "unknown_term",
            "The explicit outcome function must use identified saved GMM parameters.",
        )
    predictors = list(formula.columns)
    if set(predictors) & set(result.spec.categorical):
        fail(
            "unsupported_prediction_design",
            "General GMM outcome functions require explicit numeric input columns.",
        )
    adapter = Response(
        "nl",
        state.terms,
        {name: ("numeric", name) for name in predictors},
        {},
        {},
        {},
        "explicit_outcome_function",
        formula,
        kinds=frozenset({"xb", "stdp", "response"}),
    )
    adapter.definition = lambda kind: (
        "explicit GMM outcome function: " + function + "; saved full parameter delta covariance"
    )
    return _Model(
        result,
        state,
        None,
        "explicit",
        predictors,
        {},
        {},
        {},
        set(),
        None,
        None,
        None,
        None,
        None,
        None,
        adapter,
    )


def _standardize(model, data, alpha, batch_rows):
    from .streaming_prediction import _Replay, _Average
    from .prediction import _encode, _effect_inference

    model.state = replace(model.state, alpha=alpha)
    replay = _Replay(
        model,
        data,
        weights=True,
        batch_rows=batch_rows,
        kind="response",
        state_bytes=128 * (len(model.state.terms) + 1),
    )
    average = _Average()
    for _, frame, w in replay.batches():
        if not len(frame):
            continue
        design = _encode(model, frame)
        value = model.response_adapter.values(design, model.state.beta, "response")
        jac = model.response_adapter.jacobian(design, model.state.beta, "response")
        normalized = w / w.max()
        normalized = normalized / normalized.sum()
        average.add(torch.cat(((normalized @ value)[None], normalized @ jac)), w)
    if average.value is None:
        fail("empty_sample", "No complete positive-weight evaluation rows remain.")
    estimate = float(average.value[0])
    row = {
        "target": "standardized_outcome",
        "estimate": estimate,
        **_effect_inference(model.state, estimate, average.value[1:]),
    }
    output = pd.DataFrame([row])
    output.attrs.update(
        population="explicit supplied Dataset; fixed rows, global complete positive-weight standardization",
        source=replay.metadata(),
        delta_gradient=average.value[1:].tolist(),
        response_definition=model.response_adapter.definition("response"),
        parameter_terms=list(model.state.terms),
        uncertainty="full saved joint parameter covariance; evaluation sampling/transport uncertainty excluded",
    )
    return output


def _contrast(state, vector, alpha, target, **metadata):
    vector = torch.tensor(vector, dtype=torch.float64)
    estimate = float(vector @ state.beta)
    variance = float(vector @ state.covariance @ vector)
    if variance < -1e-10 or not math.isfinite(variance):
        fail("invalid_inference", "Saved target covariance is invalid.")
    error = math.sqrt(max(0.0, variance))
    critical = critical_value(alpha, state.df)
    return pd.DataFrame(
        [
            {
                "target": target,
                "estimate": estimate,
                "std_error": error,
                "ci_low": estimate - critical * error,
                "ci_high": estimate + critical * error,
                **metadata,
            }
        ]
    )


def _population(result, target, treatment, cohort, event, alpha):
    state = _parameters(result)
    k = len(state.terms)
    estimator = result.spec.estimator
    if estimator == "teffects":
        levels = result.extra.get("levels")
        if not isinstance(levels, list) or treatment not in levels:
            fail(
                "unknown_treatment", "Population targets require an explicit saved treatment label."
            )
        level = levels.index(treatment)
        estimand = result.extra.get("estimand")
        vector = [0.0] * k
        if target == "population_effect":
            if level == 0 or estimand == "pomeans":
                if estimand != "pomeans":
                    fail(
                        "unsupported_causal_target", "Choose a noncontrol treatment for an effect."
                    )
                vector[level], vector[0] = 1.0, -1.0
            else:
                vector[level - 1] = 1.0
        elif target == "population_potential_outcome":
            if estimand == "pomeans":
                vector[level] = 1.0
            else:
                vector[-1] = 1.0
                if level:
                    vector[level - 1] = 1.0
        else:
            fail(
                "unsupported_causal_target",
                "Select population_effect or population_potential_outcome.",
            )
        return _contrast(
            state,
            vector,
            alpha,
            target,
            treatment=treatment,
            estimand=estimand,
            conditioning_treatment=result.extra.get("target_population", {}).get("treatment_level"),
        )
    if estimator == "didregress":
        if target != "population_effect":
            fail("unsupported_causal_target", "DiD stores an ATET, not a unit mean.")
        matches = [i for i, c in enumerate(result.coefficients) if c.equation == "ATET"]
        if len(matches) != 1:
            fail("prediction_state_missing", "Saved DiD ATET identity is absent.")
        vector = [float(i == matches[0]) for i in range(k)]
        return _contrast(
            state,
            vector,
            alpha,
            target,
            estimand="TWFE ATET",
            cohort_scope=result.extra.get("adoption", "saved estimation sample"),
        )
    if estimator == "eventstudy":
        if target != "event_effect" or type(event) is not int:
            fail(
                "unsupported_causal_target",
                "Event-study evaluation requires event_effect and an exact saved relative event period.",
            )
        records = result.extra.get("event_table")
        selected = [
            row
            for row in records or []
            if row.get("relative_time") == event and row.get("estimate") is not None
        ]
        if len(selected) != 1:
            fail(
                "unsupported_event_support",
                "Requested event period is not identified by the saved fit.",
            )
        row = selected[0]
        if row.get("reference"):
            return pd.DataFrame(
                [
                    {
                        "target": target,
                        "estimate": 0.0,
                        "std_error": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "event": event,
                        "reference": True,
                    }
                ]
            )
        vector = [float(term == row["term"]) for term in state.terms]
        if sum(vector) != 1:
            fail("prediction_state_missing", "Saved event coefficient identity is incomplete.")
        return _contrast(state, vector, alpha, target, event=event, binned=row.get("binned", False))
    if estimator == "csdid":
        if target not in {"simple", "dynamic", "group", "calendar"}:
            fail(
                "unsupported_causal_target",
                "Choose an identified simple/dynamic/group/calendar ATT aggregation.",
            )
        records = result.extra.get(target)
        if target == "simple":
            records = [records]
        elif target == "group":
            records = [
                row
                for row in records or []
                if row.get("period") == cohort
                or row.get("group") == cohort
                or row.get("label") == cohort
            ]
        elif target == "dynamic" and event is not None:
            records = [
                row
                for row in records or []
                if row.get("period") == event or row.get("label") == event
            ]
        if not records or any(
            not isinstance(row, dict) or type(row.get("std_error")) not in {int, float}
            for row in records
        ):
            fail(
                "prediction_state_missing",
                "This saved CSDID aggregation/support has no full joint ATT/share influence uncertainty.",
            )
        frame = pd.DataFrame(records)
        critical = critical_value(alpha, None)
        frame["ci_low"] = frame["estimate"] - critical * frame["std_error"]
        frame["ci_high"] = frame["estimate"] + critical * frame["std_error"]
        return frame
    fail("unsupported_causal_target", "This result does not identify a population effect target.")


def _rd(result, target, side, point, correction, alpha):
    state = _parameters(result)
    cutoff = result.metrics.get("cutoff")
    if result.extra.get("derivative", 0) != 0:
        fail(
            "unsupported_rd_target",
            "A derivative/kink fit needs an explicit derivative target; it is not a cutoff mean/effect.",
        )
    if target == "cutoff_side" and result.extra.get("covariates"):
        fail(
            "prediction_state_missing",
            "Adjusted one-sided outcome means require full saved covariate-adjustment nuisance covariance; the native cutoff effect remains available.",
        )
    if type(point) not in {int, float} or point != cutoff:
        fail(
            "unsupported_rd_support",
            "RD targets are identified one-sided limits at the exact saved cutoff; arbitrary unit responses are unavailable.",
        )
    if correction not in {"conventional", "bias_corrected", "robust"}:
        fail("invalid_rd_correction", "Choose conventional, bias_corrected or robust uncertainty.")
    if target == "cutoff_effect":
        if side is not None:
            fail(
                "invalid_rd_side",
                "A cutoff effect contrasts both sides; side applies only to cutoff_side.",
            )
        term = {
            "conventional": "Conventional",
            "bias_corrected": "Bias-corrected",
            "robust": "Robust",
        }[correction]
        vector = [float(name == term) for name in state.terms]
        return _contrast(
            state, vector, alpha, target, cutoff=cutoff, design=result.extra.get("design")
        )
    if target != "cutoff_side" or side not in {"left", "right"}:
        fail("invalid_rd_side", "Choose cutoff_side and explicit left/right.")
    record = result.extra.get("cutoff_evaluation")
    if not isinstance(record, dict) or record.get("version") != 1:
        fail("prediction_state_missing", "Saved RD one-sided covariance is absent.")
    estimate = result.extra["side_estimates"][side][
        "conventional" if correction == "conventional" else "bias_corrected"
    ]
    variance = record["sides"][side][
        "robust_variance" if correction == "robust" else "conventional_variance"
    ]
    if not math.isfinite(estimate) or not math.isfinite(variance) or variance < 0:
        fail("invalid_inference", "RD side covariance is not finite nonnegative.")
    error = math.sqrt(variance)
    critical = critical_value(alpha, None)
    return pd.DataFrame(
        [
            {
                "target": target,
                "side": side,
                "cutoff": cutoff,
                "estimate": estimate,
                "std_error": error,
                "ci_low": estimate - critical * error,
                "ci_high": estimate + critical * error,
                "main_bandwidth": result.metrics["h_" + side],
                "bias_bandwidth": result.metrics["b_" + side],
            }
        ]
    )


def causal_evaluate(
    result,
    data=None,
    *,
    target,
    population="estimation",
    treatment=None,
    cohort=None,
    event=None,
    side=None,
    point=None,
    correction="robust",
    outcome_function=None,
    alpha=0.05,
    batch_rows=None,
):
    """Explicit causal population, RD cutoff or supplied GMM outcome targets.

    population='estimation' uses original full influence/sandwich covariance.
    population='fixed_evaluation' permits conditional outcome-model predictions
    or global standardization; it does not assert a transported ATE/ATT.
    """
    if not isinstance(result, ResultBundle):
        fail("invalid_result", "Supply a saved fitted ResultBundle.")
    result = deepcopy(result)
    alpha = _alpha(alpha)
    estimator = result.spec.estimator
    selectors = {
        "treatment": treatment,
        "cohort": cohort,
        "event": event,
        "side": side,
        "point": point,
        "outcome_function": outcome_function,
    }
    allowed = {
        "teffects": {"treatment"},
        "eventstudy": {"event"},
        "csdid": ({"cohort"} if target == "group" else {"event"} if target == "dynamic" else set()),
        "rdrobust": {"side", "point"},
        "gmm": {"outcome_function"},
    }.get(estimator, set())
    if any(value is not None and key not in allowed for key, value in selectors.items()):
        fail(
            "unsupported_causal_support",
            "Treatment/cohort/event/cutoff/function selectors must belong to this explicit saved target.",
        )
    metadata = {
        "estimator": result.spec.estimator,
        "target": target,
        "population": population,
        "model_id": result.id,
        "refitted": False,
    }
    with torch.device("cpu"), torch.inference_mode(False):
        try:
            if population == "fixed_evaluation":
                if not isinstance(data, Dataset) or target not in {
                    "outcome_mean",
                    "standardized_outcome",
                }:
                    fail(
                        "unsupported_causal_population",
                        "Fixed evaluation requires an explicit Dataset and outcome_mean/standardized_outcome target; it is not a transported effect.",
                    )
                model = (
                    _gmm_model(result, outcome_function)
                    if result.spec.estimator == "gmm"
                    else _nuisance_model(result, treatment)
                    if result.spec.estimator == "teffects"
                    else None
                )
                if model is None:
                    fail(
                        "unsupported_causal_target",
                        "This causal effect has no saved unit outcome-model target.",
                    )
                if target == "standardized_outcome":
                    frame = _standardize(model, data, alpha, batch_rows)
                    metadata.update(frame.attrs)
                    return materialize_predictions(iter([frame]), metadata)
                from .streaming_prediction import predict_dataset

                output = predict_dataset(
                    model,
                    data,
                    kind="response",
                    alpha=alpha,
                    interval="mean",
                    term=None,
                    outcome=treatment
                    if result.spec.estimator == "teffects"
                    else "explicit_function",
                    batch_rows=batch_rows,
                )
                output.metadata["analysis"].update(metadata)
                return output
            if population != "estimation" or data is not None:
                fail(
                    "unsupported_causal_population",
                    "Original identified effects use population=estimation without new evaluation rows.",
                )
            frame = (
                _rd(result, target, side, point, correction, alpha)
                if result.spec.estimator == "rdrobust"
                else _population(result, target, treatment, cohort, event, alpha)
            )
            metadata.update(
                uncertainty="saved full estimator influence/sandwich covariance; one alternative target at a time",
                source_hash=result.provenance.get(
                    "data_hash", result.provenance.get("sample_hash")
                ),
                support="saved identified treatment/cohort/event/cutoff only",
            )
            return materialize_predictions(iter([frame]), metadata)
        except AnalysisError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise AnalysisError(
                "prediction_state_missing",
                "Saved causal/RD evaluation identity, covariance or support metadata are incomplete.",
            ) from exc
