"""Explicit history/origin targets; native bounded state, never refitting.

Forecasts, smoothed diagnostics and transformed equations are distinct targets.
The existing native forecast uncertainty is preserved and named; requesting an
unimplemented uncertainty/path returns a structured error instead of zero bands.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle
from openecon.resources import plan_workspace
from openecon.econometrics.ordered_replay import OrderedReplay
from openecon.econometrics.postest.inference import _parameters, _alpha
from openecon.econometrics.postest.streaming_prediction import materialize_predictions


TARGETS = {
    "arima": {"forecast_levels"},
    "arch": {"forecast_levels", "conditional_variance"},
    "var": {"forecast_levels"},
    "vec": {"forecast_levels"},
    "ucm": {"forecast_levels", "components_smoothed"},
    "mswitch": {"probabilities_filtered", "probabilities_predicted", "probabilities_smoothed"},
    "ardl": {"fitted_levels"},
    "nardl": {"fitted_levels"},
    "ahreg": {"fitted_transformed"},
    "xtdpd": {"fitted_transformed"},
}


def fail(code, message):
    raise AnalysisError(code, message)


def _history(result, history, origin):
    if not isinstance(history, Dataset):
        fail(
            "prediction_history_required",
            "Supply an explicit Dataset history, including all required lag/filter inputs.",
        )
    if origin == "end":
        return history
    if (
        result.spec.time is None
        or type(origin) not in {int, float}
        or not math.isfinite(origin)
        or origin != int(origin)
    ):
        fail(
            "invalid_prediction_origin",
            "Use origin='end' or an exact integer period in the declared time column.",
        )
    name = result.spec.time

    def frames():
        found = False
        for block in history.iter_batches():
            if name not in block:
                fail("missing_columns", "History lacks its declared time column.")
            if not pd.api.types.is_numeric_dtype(block[name].dtype):
                fail("invalid_prediction_origin", "Numeric origins require integer-period history.")
            found |= bool((block[name] == origin).any())
            selected = block.loc[block[name] <= origin]
            if len(selected):
                yield selected
        if not found:
            fail(
                "invalid_prediction_origin",
                "The explicit forecast origin is absent from the history.",
            )

    return Dataset.from_batches(frames, history.columns)


def _future(future, horizon):
    if not isinstance(future, Dataset):
        return future
    width = len(future.columns)
    plan_workspace("bounded future path", {"future_input": 256 * horizon * max(1, width)})
    blocks, count = [], 0
    for block in future.iter_batches(batch_rows=min(horizon + 1, 65536)):
        count += len(block)
        if count > horizon:
            fail("invalid_exog", "Future path must have exactly one row per horizon step.")
        blocks.append(block)
    if count != horizon:
        fail("invalid_exog", "Future path must have exactly one row per horizon step.")
    return pd.concat(blocks, ignore_index=True)


def _saved_coding(result, ordered):
    from .linear_prediction import _coding
    from openecon.streaming_design import _label

    _, categories = _coding(result, result.spec.categorical)
    for name, levels in categories.items():
        allowed = {_label(value) for value in levels}
        if any(_label(value) not in allowed for value in ordered.category_dtypes[name].categories):
            fail(
                "unknown_category",
                "History includes a categorical level absent from the saved fit.",
            )
        record = result.provenance["categorical_encoding"][name]
        ordered.category_dtypes[name] = pd.CategoricalDtype(
            levels, ordered=record.get("ordered", False)
        )


def _arima_state(result, ordered):
    from openecon.econometrics.streaming_arima import _ARIMAReplay

    replay = _ARIMAReplay(ordered)
    theta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    k, orders = replay.k, replay.orders
    expected = [*replay.design.terms]
    for label, p, q in [
        ("ARMA", orders.p, orders.q),
        (f"ARMA{orders.period}", orders.seasonal_p, orders.seasonal_q),
    ]:
        expected += [f"{label}:L{i}.ar" for i in range(1, p + 1)]
        expected += [f"{label}:L{i}.ma" for i in range(1, q + 1)]
    expected += ["/sigma"]
    if expected != [c.term for c in result.coefficients]:
        fail("prediction_state_missing", "ARIMA history does not reproduce the saved design/order.")
    units = torch.diag(
        torch.cat(
            (
                replay.scale_y / replay.scale_x,
                torch.ones(orders.n_arma, dtype=torch.float64),
                torch.tensor([replay.scale_y], dtype=torch.float64),
            )
        )
    )
    reporting = theta.clone()
    if replay.constant:
        units[0, 1:k] = -replay.scale_y * replay.mean_x[1:] / replay.scale_x[1:]
        reporting[0] -= replay.mean_y
    working = torch.linalg.solve(units, reporting)
    decomposition = replay.decomposition(working)
    result.extra["last_state"] = replay.state_record(theta, decomposition)
    phi, ma, sphi, sma = replay.like.split(working[:-1])
    result.extra.update(ar=phi, ma=ma, seasonal={"ar": sphi, "ma": sma, "period": orders.period})


def _arch_state(result, ordered, stack):
    from openecon.econometrics.streaming_arch import _Replay

    replay = _Replay(ordered)
    stack.callback(replay.like.close)
    layout = replay.layout
    terms, _ = layout.terms(
        result.spec.outcome, replay.design.terms, replay.zdesign.terms[1:] if replay.zdesign else []
    )
    if terms != [c.term for c in result.coefficients]:
        fail(
            "prediction_state_missing",
            "ARCH history does not reproduce the fitted mean/variance layout.",
        )
    theta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    final = replay.like.evaluate(torch.linalg.solve(replay.back, theta), derivatives=False)
    if final is None:
        fail("prediction_domain", "The saved ARCH recursion is undefined on this history.")
    result.extra["state"] = {
        "residuals": final.tail_residual[-layout.max_lag :].tolist(),
        "disturbances": final.tail_disturbance[-layout.max_lag :].tolist(),
        "variances": final.tail_variance[-layout.max_lag :].tolist(),
        "presample_variance": final.presample_variance,
        "last_period": ordered.last_period,
    }


def _var_state(result, ordered):
    if result.spec.estimator == "var":
        names = result.extra["layout"]["variables"]
        p = result.extra["layout"]["lags"]
    else:
        names = result.extra["layout"]["variables"]
        p = len(result.extra["var_representation"]["A"])
    if ordered.count < p:
        fail("prediction_history_required", "History has fewer levels than the fitted lag order.")
    tail = None
    for block in ordered.source.iter_batches(batch_rows=ordered.rows):
        tail = block[names] if tail is None else pd.concat((tail, block[names]), ignore_index=True)
        tail = tail.tail(p).copy()
    record = result.extra["forecast"]
    trend = ordered.count
    if ordered.last_period is not None and record.get("last_period") is not None:
        trend = int(record["last_trend"]) + ordered.last_period - int(record["last_period"])
    record.update(
        last_values=tail.to_numpy(dtype=float).tolist(),
        last_period=ordered.last_period,
        last_trend=trend,
    )


def _forecast(result, ordered, horizon, future, alpha, stack):
    from openecon.econometrics.arima.forecast import forecast

    estimator = result.spec.estimator
    if estimator == "arima":
        _arima_state(result, ordered)
    elif estimator == "arch":
        _arch_state(result, ordered, stack)
    elif estimator in {"var", "vec"}:
        _var_state(result, ordered)
    elif estimator == "ucm":
        from openecon.econometrics.streaming_ucm import ucm_forecast_streaming

        return ucm_forecast_streaming(
            result, horizon, data=_projected(ordered), exog=future, alpha=alpha
        )
    return forecast(result, horizon, exog=future, alpha=alpha)


def _projected(ordered):
    return Dataset.from_batches(
        lambda: ordered.source.iter_batches(ordered.columns, batch_rows=ordered.rows),
        ordered.columns,
        row_count=ordered.count,
    )


def _fitted_ardl(result, ordered, alpha):
    from openecon.econometrics.streaming_ardl import _ARDLReplay
    from openecon.econometrics.streaming_nardl import _NARDLReplay
    from openecon.engines.inference import critical_value

    state = _parameters(result)
    replay = _NARDLReplay(ordered) if result.spec.estimator == "nardl" else _ARDLReplay(ordered)
    names = result.extra.get("expanded_predictors", result.spec.predictors)
    levels = result.extra.get("levels_coefficients")
    if levels is None and not result.extra.get("ec"):
        levels = {c.term: c.estimate for c in result.coefficients}
        levels.update(result.extra.get("constrained_terms", {}))
        covariance = state.covariance.tolist()
        if len(levels) != len(state.terms):
            fail(
                "prediction_state_missing",
                "Fixed level constraints require a saved full level covariance.",
            )
    else:
        covariance = result.extra.get("levels_covariance")
    if not isinstance(levels, dict) or covariance is None:
        fail(
            "prediction_state_missing",
            "Saved EC/NARDL prediction needs full level coefficients and covariance.",
        )
    terms = list(levels)
    beta, covariance = (
        torch.tensor(list(levels.values()), dtype=torch.float64),
        torch.tensor(covariance, dtype=torch.float64),
    )
    if covariance.shape != (len(terms), len(terms)) or not bool(torch.isfinite(covariance).all()):
        fail("invalid_result", "Saved ARDL level covariance is invalid.")
    scale = max(1.0, float(covariance.abs().max()))
    if (
        len(terms) > 384
        or not torch.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)
        or float(torch.linalg.eigvalsh((covariance + covariance.T) / 2).min()) < -1e-10 * scale
    ):
        fail(
            "invalid_result",
            "Saved ARDL level covariance must be bounded, symmetric and positive semidefinite.",
        )
    lags = result.extra.get("lags")
    if not isinstance(lags, dict) or set(lags) != {result.spec.outcome, *names}:
        fail("prediction_state_missing", "Saved selected lag orders are absent or inconsistent.")
    replay.names = names
    replay.maxlags = [lags[result.spec.outcome], *[lags[name] for name in names]]
    if any(type(x) is not int or x < 0 for x in replay.maxlags):
        fail("invalid_result", "Saved lag orders must be nonnegative integers.")
    replay.largest = max(replay.maxlags)
    replay.trend = result.extra["trend"]
    exog = result.spec.columns.get("exog") or []
    replay.exog = [exog] if isinstance(exog, str) else list(exog)
    plan_workspace(
        "saved ARDL lag design",
        {
            "bounded_lag_carry_and_design": 256
            * (ordered.rows + replay.largest)
            * max(1, len(terms))
        },
    )
    critical = critical_value(alpha, state.df)
    for block in replay.factory():
        x = replay.matrix(block, terms)
        mean = x @ beta
        variance = torch.einsum("nk,kl,nl->n", x, covariance, x)
        if not bool(torch.isfinite(mean).all()) or bool((variance < -1e-10).any()):
            fail("prediction_domain", "ARDL saved predictions/covariance are invalid on this path.")
        error = variance.clamp_min(0).sqrt()
        yield pd.DataFrame(
            {
                "physical_row": block[ordered.positions_column],
                "fitted_level": mean.tolist(),
                "std_error": error.tolist(),
                "ci_low": (mean - critical * error).tolist(),
                "ci_high": (mean + critical * error).tolist(),
            }
        )


def _panel(result, history, alpha, batch_rows, metadata):
    from openecon.econometrics.replay_sample import ReplaySample
    from openecon.econometrics.streaming_dpanel import _PanelStore
    from openecon.engines.inference import critical_value

    state = _parameters(result)
    with ExitStack() as stack:
        if result.spec.estimator == "ahreg":
            sample = ReplaySample(result.spec, history, batch_rows=batch_rows).prepare()
            sample.plan_rows(
                "saved AH transformed equations",
                {"panel_SQLite_cache": 4 * 1024**2},
                256 * (len(sample.columns) + 8),
            )
            store = _PanelStore(sample, [result.spec.outcome, *result.spec.predictors])
            stack.callback(store.close)
            store.seed()
            store.ah_equations(result.spec.options.get("instrument", "levels"))
            expected = [f"L1.{result.spec.outcome}", *result.spec.predictors]
            if list(state.terms) != expected:
                fail(
                    "prediction_state_missing",
                    "Saved AH transformed coefficient mapping is unavailable.",
                )

            def rows():
                for _, period, values, positions in store.blocks():
                    yield (
                        torch.cat((values[:, 1:2], values[:, 3:]), 1),
                        positions,
                        period,
                        "first_difference",
                    )
        else:
            from openecon.econometrics import streaming_dynamic_gmm as core

            opts = core.model.read_options(result.spec)
            private = result.spec.model_copy(
                update={
                    "columns": {**result.spec.columns, "replay_instrument_inputs": opts.columns}
                }
            )
            sample = ReplaySample(private, history, batch_rows=batch_rows).prepare()
            sample.spec = result.spec
            sample.plan_rows(
                "saved dynamic-panel transformed equations",
                {"panel_SQLite_cache": 4 * 1024**2},
                256 * (len(sample.columns) + 8),
            )
            names = list(
                dict.fromkeys([result.spec.outcome, *result.spec.predictors, *opts.columns])
            )
            store = core._Store(sample, names)
            stack.callback(store.close)
            store.seed()
            store.discover(opts)
            layout = core._metadata(store, opts)
            if set(state.terms) - set(layout[3]):
                fail(
                    "prediction_state_missing",
                    "Saved panel transformed term/time-dummy mapping is absent.",
                )
            selected = [layout[3].index(term) for term in state.terms]

            def rows():
                for _, current, positions in core._panels(store, opts, layout, keep_x=selected):
                    # Only the transformed equation is exposed. The centered
                    # system-level equation is not a unit-effect forecast.
                    x = current.x[: current.n_t]
                    chosen = current.obs.rows[current.transform.source]
                    physical = [positions[int(i)] for i in chosen]
                    yield (
                        x,
                        physical,
                        current.transform.period,
                        "FOD" if opts.orthogonal else "first_difference",
                    )

        critical = critical_value(alpha, state.df)
        for x, positions, period, definition in rows():
            mean = x @ state.beta
            variance = torch.einsum("nk,kl,nl->n", x, state.covariance, x)
            if not bool(torch.isfinite(mean).all()) or bool((variance < -1e-10).any()):
                fail("prediction_domain", "Saved transformed panel covariance is invalid.")
            error = variance.clamp_min(0).sqrt()
            yield pd.DataFrame(
                {
                    "physical_row": positions,
                    "period": period.tolist(),
                    "transformation": definition,
                    "fitted_transformed": mean.tolist(),
                    "std_error": error.tolist(),
                    "ci_low": (mean - critical * error).tolist(),
                    "ci_high": (mean + critical * error).tolist(),
                }
            )
        for _ in sample.batches():
            pass
        metadata.update(
            history_hash=sample.provenance()["data_hash"],
            resource_plan=sample.adapter_plan.record(),
        )


def dynamic_predict(
    result,
    history,
    *,
    target,
    origin,
    horizon=None,
    future=None,
    uncertainty="native",
    alpha=0.05,
    batch_rows=None,
):
    """Owned Dataset for explicit dynamic targets, with no implicit static mean.

    origin='end' consumes supplied history; integer origin excludes later rows.
    ARIMA/ARCH/UCM/VEC native forecast intervals contain innovation/filter error;
    VAR additionally uses its recorded native parameter term when available.
    Parameter-only dynamic forecast intervals are currently rejected. ARDL and
    dynamic-panel fitted targets use the saved full parameter delta covariance.
    """
    if not isinstance(result, ResultBundle) or target not in TARGETS.get(
        result.spec.estimator, set()
    ):
        fail(
            "unsupported_dynamic_target",
            "This family/target has no identified saved-history adapter.",
        )
    if uncertainty != "native":
        fail(
            "unsupported_dynamic_uncertainty",
            "Request native uncertainty; parameter-only/combined paths require a separately validated tangent/filter contract.",
        )
    if batch_rows is not None and (type(batch_rows) is not int or not 1 <= batch_rows <= 65536):
        fail("invalid_batch_size", "batch_rows must be an integer from 1 to 65536.")
    alpha = _alpha(alpha)
    result = deepcopy(result)
    _parameters(result)
    history = _history(result, history, origin)
    forecast_target = target in {"forecast_levels", "conditional_variance"}
    if forecast_target:
        if type(horizon) is not int or not 1 <= horizon <= 5000:
            fail("invalid_prediction_horizon", "Supply an explicit integer horizon from 1 to 5000.")
    elif horizon is not None or future is not None:
        fail(
            "invalid_prediction_horizon", "Horizon/future inputs apply only to recursive forecasts."
        )
    future = _future(future, horizon) if forecast_target else None
    metadata = {
        "estimator": result.spec.estimator,
        "target": target,
        "origin": origin,
        "horizon": horizon,
        "uncertainty": "native family contract; see uncertainty_definition",
        "full_source_collected": False,
        "model_id": result.id,
        "refitted": False,
    }
    try:
        with torch.device("cpu"), torch.no_grad(), ExitStack() as stack:
            if result.spec.estimator in {"ahreg", "xtdpd"}:
                if origin != "end":
                    fail(
                        "unsupported_dynamic_origin",
                        "Panel transformed targets require explicit complete histories through end.",
                    )
                metadata["uncertainty_definition"] = (
                    "full saved parameter covariance on the transformed equation; no unit effects or level forecast"
                )
                return materialize_predictions(
                    _panel(result, history, alpha, batch_rows, metadata), metadata
                )
            ordered = stack.enter_context(OrderedReplay(result.spec, history))
            _saved_coding(result, ordered)
            if batch_rows:
                ordered.rows = min(ordered.rows, batch_rows)
            if result.spec.estimator == "nardl":
                expected = result.provenance.get("original_data_hash")
                if not expected or ordered.raw_hash != expected:
                    fail(
                        "prediction_history_required",
                        "NARDL fitted partial sums need the original complete source and its zero origin; an arbitrary truncated path is not interchangeable.",
                    )
            metadata.update(
                history_hash=ordered.raw_hash,
                ordered_history_rows=ordered.count,
                last_period=ordered.last_period,
                resource_plan=ordered.plan.record(),
            )
            if forecast_target:
                output = _forecast(result, ordered, horizon, future, alpha, stack)
                if target == "conditional_variance":
                    output = output[["period", "variance_forecast"]].copy()
                    metadata["uncertainty_definition"] = (
                        "native expected conditional-variance recursion; no uncertainty interval for variance state"
                    )
                else:
                    metadata["uncertainty_definition"] = output.attrs.get(
                        "mse_formula",
                        output.attrs.get(
                            "standard_errors",
                            "native innovation/filter forecast error; parameters fixed unless family metadata says otherwise",
                        ),
                    )
                frames = iter([output])
            elif target == "fitted_levels":
                metadata["uncertainty_definition"] = (
                    "full saved level parameter covariance conditional on the supplied lag/partial-sum path"
                )
                frames = _fitted_ardl(result, ordered, alpha)
            else:
                if result.spec.estimator == "ucm":
                    from openecon.econometrics.tsmodels.ucm import ucm_components

                    output = ucm_components(result, _projected(ordered))
                    metadata["uncertainty_definition"] = (
                        "native smoothed latent-state covariance; parameters fixed"
                    )
                else:
                    from openecon.econometrics.tsmodels.mswitch import mswitch_probabilities

                    output = mswitch_probabilities(result, _projected(ordered))
                    metadata["uncertainty_definition"] = (
                        "native Hamilton/Kim regime probabilities conditional on saved parameters; no probability parameter bands"
                    )

                def frames_from_output():
                    for block in output.iter_batches(batch_rows=ordered.rows):
                        if result.spec.estimator == "mswitch":
                            prefix = target.removeprefix("probabilities_") + "_state"
                            block = block[
                                [
                                    name
                                    for name in block.columns
                                    if name == "period" or name.startswith(prefix)
                                ]
                            ]
                        yield block

                frames = frames_from_output()

            def verified():
                yield from frames
                ordered.verify_original()

            return materialize_predictions(verified(), metadata)
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "prediction_state_missing",
            "Saved dynamic parameters/state cannot reproduce this explicit target.",
        ) from exc
