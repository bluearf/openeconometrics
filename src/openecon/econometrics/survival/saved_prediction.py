"""Explicit survival targets and complete, immutable disk Cox baselines.

All intervals here are saved-parameter delta intervals. Cox baseline counting
process variation is not included; the output records that distinction.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
import math
import numbers

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle
from openecon.econometrics.postest.advanced_prediction import Response
from openecon.econometrics.postest.inference import _parameters, _alpha
from openecon.econometrics.postest.linear_prediction import _coding
from openecon.econometrics.postest.streaming_prediction import materialize_predictions


def fail(code, message):
    raise AnalysisError(code, message)


def _stratum_key(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
    return json.dumps(value, ensure_ascii=False)


def _result(result, estimator):
    if not isinstance(result, ResultBundle) or result.spec.estimator != estimator:
        fail("unsupported_result", f"This target needs a saved {estimator} result.")
    return deepcopy(result)


def cox_baseline(result, data, *, batch_rows=None):
    """Full failure-time baseline as an owned Dataset, never the display preview.

    Requires the original Dataset fit's projected source hash. Saved JSON works;
    legacy resident fits use stcurve(data=resident_frame) instead. Gradients use
    coefficient order and include the baseline's dependence on saved beta.
    """
    from openecon.econometrics import streaming_cox as core

    result = _result(result, "stcox")
    if not isinstance(data, Dataset):
        fail("invalid_data", "Full disk baseline needs the original estimation Dataset.")
    if result.spec.columns.get("tvc"):
        fail(
            "unsupported_survival_path",
            "Cox TVC survival needs an explicit event-time covariate path.",
        )
    expected = result.extra.get("replay_risk_sets", {}).get("source_spool_hash")
    if not expected or not result.provenance.get("data_hash"):
        fail(
            "prediction_state_missing",
            "Full disk baseline needs the original replay source identity.",
        )
    state = _parameters(result)
    with torch.device("cpu"), ExitStack() as stack:
        sample, _, store, terms, transform, centre, resource, _, _ = core._prepare(
            result.spec, data, batch_rows, stack
        )
        if (
            sample.provenance()["data_hash"] != result.provenance["data_hash"]
            or sample.provenance()["sample_positions_hash"]
            != result.provenance.get("sample_positions_hash")
            or set(terms) != set(state.terms)
        ):
            fail(
                "data_mismatch",
                "The baseline data differ from the saved estimation source/sample/design.",
            )
        order = [state.terms.index(term) for term in terms]
        beta = torch.linalg.solve(transform, state.beta[order])
        core._Objective(
            store, "efron" if result.spec.options.get("ties") == "efron" else "breslow"
        )(beta, record=True, shift=float(centre @ beta))

        def frames():
            cumulative = torch.zeros(len(terms), dtype=torch.float64)
            previous, position = None, 0
            pending = []
            query = (
                "SELECT e.s,e.t,e.increment,e.hazard,e.kp,e.pull,e.a,s.label "
                "FROM events e JOIN strata s ON s.s=e.s ORDER BY e.s,e.t"
            )
            for s, t, jump, hazard, kp, pull, a, label in store.connection.execute(query):
                if s != previous:
                    cumulative.zero_()
                    previous = s
                gradient = -jump * (core._unpack(pull, len(terms)) / a + centre)
                cumulative += torch.linalg.solve(transform.T, gradient)
                row = {
                    "stratum": json.loads(label),
                    "time": t,
                    "hazard_jump": jump,
                    "cumulative_hazard": hazard,
                    "survivor_kp": math.exp(kp),
                }
                for j, original in enumerate(order):
                    row[f"gradient[{original}]"] = float(cumulative[j])
                    row[f"jump_gradient[{original}]"] = float(
                        torch.linalg.solve(transform.T, gradient)[j]
                    )
                pending.append(row)
                if len(pending) == sample.rows:
                    yield pd.DataFrame(pending, index=range(position, position + len(pending)))
                    position += len(pending)
                    pending.clear()
            if pending:
                yield pd.DataFrame(pending, index=range(position, position + len(pending)))
            if store.digest() != store.checksum:
                fail(
                    "cox_spill_failed", "Immutable Cox risk records changed during baseline replay."
                )
            for _ in sample.batches():
                pass

        output = materialize_predictions(
            frames(),
            {
                "estimator": "stcox",
                "target": "full_baseline",
                "full_step_function": True,
                "thinned": False,
                "terms": list(state.terms),
                "model_id": result.id,
                "parameters": state.beta.tolist(),
                "source_hash": result.provenance["data_hash"],
                "ties": result.spec.options.get("ties", "breslow"),
                "resource_plan": resource,
                "uncertainty": "gradient of full baseline with respect to saved parameters; no counting-process variance",
            },
        )
        return output


class _GammaLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, time, mu, a, kappa, density):
        from .parametric import SurvivalData
        from .ggamma import ggamma_pieces

        piece = ggamma_pieces(
            SurvivalData(time, None, torch.full_like(time, float(density))), mu, a, kappa
        )
        ctx.save_for_backward(*piece.grad)
        return piece.value

    @staticmethod
    def backward(ctx, gradient):
        gm, ga, gk = ctx.saved_tensors
        return None, gradient * gm, gradient * ga, (gradient * gk).sum(), None


class Parametric(Response):
    time: object = None
    probability: float = 0.5

    def required(self):
        return [self.time] if isinstance(self.time, str) else []

    def _time(self, design):
        t = (
            self.column(design, self.time)
            if isinstance(self.time, str)
            else torch.full_like(design.deterministic, self.time)
        )
        if bool((t <= 0).any()) or not bool(torch.isfinite(t).all()):
            fail(
                "invalid_survival_time",
                "Survival evaluation times must be finite and strictly positive.",
            )
        return t

    def logterms(self, t, mu, a, kappa=None):
        from .parametric import closed_form, SurvivalData

        dist, metric = self.extra["distribution"], self.extra["metric"]
        if dist == "ggamma":
            return (_GammaLog.apply(t, mu, a, kappa, True), _GammaLog.apply(t, mu, a, kappa, False))
        logf = closed_form(dist, metric, SurvivalData(t, None, torch.ones_like(t)), mu, a).value
        logs = closed_form(dist, metric, SurvivalData(t, None, torch.zeros_like(t)), mu, a).value
        return logf, logs

    def _values(self, design, beta, kind):
        mu = self.index(design, beta, "outcome") + design.deterministic
        if kind in {"xb", "stdp"}:
            return mu
        dist, metric = self.extra["distribution"], self.extra["metric"]
        a = None if dist == "exponential" else self.index(design, beta, "ancillary")
        kappa = self.parameter(beta, "/kappa") if dist == "ggamma" else None
        if self.target == "relative_hazard":
            if metric != "ph":
                fail(
                    "unsupported_survival_target",
                    "Relative hazard requires the PH parameterization.",
                )
            return mu.exp()
        if self.target in {"survival", "hazard", "cumulative_hazard"}:
            logf, logs = self.logterms(self._time(design), mu, a, kappa)
            return (
                logs.exp()
                if self.target == "survival"
                else -logs
                if self.target == "cumulative_hazard"
                else (logf - logs).exp()
            )
        if self.target == "mean":
            if dist == "exponential":
                return (-mu if metric == "ph" else mu).exp()
            if dist == "weibull":
                p = a.exp()
                return (
                    (-mu / p if metric == "ph" else mu) + torch.lgamma(1 + p.reciprocal())
                ).exp()
            if dist == "lognormal":
                return (mu + 0.5 * (2 * a).exp()).exp()
            if dist == "loglogistic":
                gamma = a.exp()
                if bool((gamma >= 1).any()):
                    fail(
                        "undefined_survival_mean",
                        "The loglogistic mean exists only when gamma < 1.",
                    )
                return mu.exp() * math.pi * gamma / torch.sin(math.pi * gamma)
            if dist == "ggamma":
                if abs(float(kappa.detach())) < 1e-10:
                    # The kappa derivative is the continuous lognormal limit.
                    sigma = a.exp()
                    return (mu + sigma.square() / 2).exp() * (
                        1 - kappa * (sigma / 2 + sigma.pow(3) / 6)
                    )
                if abs(float(kappa.detach())) < 0.02:
                    fail(
                        "prediction_precision",
                        "Generalized-gamma mean near kappa=0 requires a validated gamma-ratio expansion.",
                    )
                g, exponent = kappa.square().reciprocal(), a.exp() / kappa
                if bool((g + exponent <= 0).any()):
                    fail(
                        "undefined_survival_mean", "This generalized-gamma tail has no finite mean."
                    )
                return (
                    mu - exponent * g.log() + torch.lgamma(g + exponent) - torch.lgamma(g)
                ).exp()
            fail(
                "unsupported_survival_target",
                "Gompertz mean is not implemented; negative shape can imply cure probability.",
            )
        q = -math.log1p(-self.probability)
        if dist in {"exponential", "weibull"}:
            p = torch.ones_like(mu) if a is None else a.exp()
            return ((math.log(q) - mu) / p if metric == "ph" else mu + math.log(q) / p).exp()
        if dist == "lognormal":
            z = torch.special.ndtri(torch.tensor(self.probability, dtype=torch.float64))
            return (mu + a.exp() * z).exp()
        if dist == "loglogistic":
            return (mu + a.exp() * math.log(self.probability / (1 - self.probability))).exp()
        if dist == "gompertz":
            z = a * q * (-mu).exp()
            if bool((z <= -1).any()):
                fail(
                    "undefined_survival_quantile",
                    "Requested quantile lies in the Gompertz cure mass.",
                )
            small = a.abs() < 1e-7
            safe = torch.where(small, torch.ones_like(a), a)
            return torch.where(
                small, q * (-mu).exp() - a * (q * (-mu).exp()).square() / 2, torch.log1p(z) / safe
            )
        # Native inverse in log time; implicit gradient from the native survival
        # kernel. The detached search does not allocate a row-by-row history.
        with torch.no_grad():
            scale = a.exp()
            lo, hi = mu - 40 * scale, mu + 40 * scale
            from .ggamma import _log_terms

            wanted = math.log1p(-self.probability)
            for _ in range(90):
                mid = (lo + hi) / 2
                _, logs = _log_terms(mid, mu, a, float(kappa.detach()))
                high = logs < wanted
                hi, lo = torch.where(high, mid, hi), torch.where(high, lo, mid)
            t = ((lo + hi) / 2).exp()
            _, final = _log_terms(t.log(), mu, a, float(kappa.detach()))
            if not bool(torch.isfinite(t).all()) or not bool((final - wanted).abs().lt(1e-8).all()):
                fail(
                    "prediction_precision",
                    "The generalized-gamma quantile cannot be resolved in float64.",
                )
        logf, logs = self.logterms(t, mu, a, kappa)
        return t + (logs - logs.detach()) / (logf - logs).detach().exp()


def _parametric_model(result, target, time, probability):
    from .streg import ANCILLARY, resolve_metric
    from openecon.econometrics.postest.prediction import _Model

    result = _result(result, "streg")
    state = _parameters(result)
    dist = result.extra.get("distribution")
    metric = result.extra.get("metric")
    if dist != result.spec.options.get("distribution") or metric != resolve_metric(
        dist, result.spec.options.get("metric")
    ):
        fail(
            "prediction_state_missing",
            "Saved survival distribution/metric do not reproduce the specification.",
        )
    allowed = {"survival", "hazard", "cumulative_hazard", "relative_hazard", "mean", "quantile"}
    if target not in allowed:
        fail(
            "unsupported_survival_target",
            "Choose an explicit survival target: " + ", ".join(sorted(allowed)) + ".",
        )
    if target in {"survival", "hazard", "cumulative_hazard"}:
        if not isinstance(time, str) and (
            type(time) not in {int, float} or not math.isfinite(time) or time <= 0
        ):
            fail("invalid_survival_time", "Supply a positive time or evaluation time column.")
    elif time is not None:
        fail(
            "invalid_survival_time", "Time applies only to survival, hazard and cumulative hazard."
        )
    if type(probability) not in {int, float} or not 0 < probability < 1:
        fail("invalid_survival_quantile", "quantile must be strictly between zero and one.")
    spec = result.spec
    strata = spec.columns.get("strata") or []
    strata = [strata] if isinstance(strata, str) else list(strata)
    anc_names = spec.columns.get("ancillary") or []
    anc_names = [anc_names] if isinstance(anc_names, str) else list(anc_names)
    predictors = list(dict.fromkeys([*spec.predictors, *anc_names, *strata]))
    result.spec = spec.model_copy(
        update={"intercept": True, "categorical": list(dict.fromkeys([*spec.categorical, *strata]))}
    )
    features, categories = _coding(result, predictors)
    raw = {key: value for key, value in features.items() if value[0] != "constant"}
    if isinstance(time, str):
        raw.setdefault(time, ("numeric", time))

    def basis(names, prefix=""):
        return {
            prefix + key: key
            for key, (kind, value) in features.items()
            if key == "Intercept"
            or (kind == "numeric" and value in names)
            or (kind == "category" and value[0] in names)
        }

    equations = {"outcome": basis([*spec.predictors, *strata])}
    anc = ANCILLARY.get(dist)
    if anc:
        equations["ancillary"] = (
            {f"/{anc}": "Intercept"}
            if not anc_names and not strata
            else basis([*anc_names, *strata], anc + ":")
        )
    expected = {term for equation in equations.values() for term in equation} | (
        {"/kappa"} if dist == "ggamma" else set()
    )
    omitted = set(result.provenance.get("omitted_terms", []))
    if set(state.terms) - expected or expected - set(state.terms) - omitted:
        fail(
            "prediction_state_missing",
            "Saved survival main/ancillary coefficient mapping is incomplete.",
        )
    equations = {
        key: {term: feature for term, feature in value.items() if term in state.terms}
        for key, value in equations.items()
    }
    adapter = Parametric(
        "streg",
        state.terms,
        raw,
        equations,
        {},
        result.extra,
        target,
        kinds=frozenset({"xb", "stdp", "response"}),
    )
    adapter.time, adapter.probability = time, float(probability)
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
        spec.columns.get("offset"),
        None,
        None,
        None,
        None,
        None,
        adapter,
    )


class Cox(Response):
    def required(self):
        return [x for x in (self.time, self.stratum) if isinstance(x, str)]

    def definition(self, kind):
        return (
            f"Cox {self.target}, full baseline; saved-parameter delta only; "
            "baseline counting-process variance excluded; constant evaluation covariates"
        )

    def encode(self, frame, design):
        design = super().encode(frame, design)
        values, gradients, quantiles = [], [], []
        k = len(self.terms)
        indexes = self.index(design, self.saved, "outcome") + design.deterministic
        for i, (_, row) in enumerate(frame.iterrows()):
            label = frame[self.stratum].iloc[i] if self.stratum else None
            key = _stratum_key(label)
            support = self.sql.execute("SELECT MAX(t) FROM baseline WHERE s=?", (key,)).fetchone()[
                0
            ]
            if support is None:
                fail("unknown_survival_stratum", "Evaluation stratum has no saved full baseline.")
            if self.target == "quantile":
                mu = float(indexes[i])
                needed = -math.log1p(-self.probability) * math.exp(-mu)
                found = self.sql.execute(
                    "SELECT t FROM baseline WHERE s=? AND h>=? ORDER BY t LIMIT 1", (key, needed)
                ).fetchone()
                if found is None:
                    fail(
                        "undefined_survival_quantile",
                        "The full observed Cox step function does not reach this quantile.",
                    )
                quantiles.append(found[0])
                continue
            t = row[self.time] if isinstance(self.time, str) else self.time
            if type(t) is bool or not math.isfinite(float(t)) or float(t) < 0:
                fail("invalid_survival_time", "Cox times must be finite and nonnegative.")
            if t > support:
                fail(
                    "unsupported_survival_extrapolation",
                    "Requested time exceeds the last baseline failure time in this stratum.",
                )
            query = (
                "SELECT jump,jg FROM baseline WHERE s=? AND t=?"
                if self.target == "hazard_jump"
                else "SELECT h,g FROM baseline WHERE s=? AND t<=? ORDER BY t DESC LIMIT 1"
            )
            found = self.sql.execute(query, (key, float(t))).fetchone()
            values.append(found[0] if found else 0.0)
            gradients.append(
                torch.frombuffer(bytearray(found[1]), dtype=torch.float64)
                if found
                else torch.zeros(k, dtype=torch.float64)
            )
        design.baseline_value = torch.tensor(values, dtype=torch.float64)
        design.baseline_gradient = (
            torch.stack(gradients) if gradients else torch.zeros((0, k), dtype=torch.float64)
        )
        design.quantile_value = torch.tensor(quantiles, dtype=torch.float64)
        return design

    def _values(self, design, beta, kind):
        mu = self.index(design, beta, "outcome") + design.deterministic
        if kind in {"xb", "stdp"}:
            return mu
        if self.target == "relative_hazard":
            return mu.exp()
        if self.target == "quantile":
            return design.quantile_value + beta.sum() * 0
        base = design.baseline_value + design.baseline_gradient @ (beta - self.saved)
        hazard = base * mu.exp()
        return (-hazard).exp() if self.target == "survival" else hazard


def _cox_model(result, target, time, probability, baseline, stack):
    from openecon.econometrics.postest.prediction import _Model
    from openecon.econometrics.stats.spill import Spill

    result = _result(result, "stcox")
    if result.spec.columns.get("tvc"):
        fail(
            "unsupported_survival_path",
            "Cox TVC targets require an explicit event-time covariate path.",
        )
    allowed = {"survival", "cumulative_hazard", "hazard_jump", "quantile", "relative_hazard"}
    if target not in allowed:
        fail(
            "unsupported_survival_target",
            "Cox supports "
            + ", ".join(sorted(allowed))
            + "; instantaneous hazard is unidentified by its step baseline.",
        )
    if type(probability) not in {int, float} or not 0 < probability < 1:
        fail("invalid_survival_quantile", "quantile must lie strictly between zero and one.")
    state = _parameters(result)
    if (
        target not in {"relative_hazard", "quantile"}
        and not isinstance(time, str)
        and (type(time) not in {int, float} or not math.isfinite(time) or time < 0)
    ):
        fail("invalid_survival_time", "Cox survival requires a time or evaluation time column.")
    if target in {"relative_hazard", "quantile"} and time is not None:
        fail("invalid_survival_time", "Time does not apply to relative hazard or quantile.")
    features, categories = _coding(result, result.spec.predictors)
    expected = set(features) - set(result.provenance.get("omitted_terms", []))
    if expected != set(state.terms):
        fail("prediction_state_missing", "Cox saved design does not reproduce its parameters.")
    adapter = Cox(
        "stcox",
        state.terms,
        {k: v for k, v in features.items() if v[0] != "constant"},
        {"outcome": {name: name for name in state.terms}},
        {},
        {},
        target,
        kinds=frozenset({"xb", "stdp", "response"}),
    )
    adapter.saved, adapter.time, adapter.probability = state.beta, time, probability
    strata = result.spec.columns.get("strata")
    if isinstance(strata, list):
        if len(strata) > 1:
            fail(
                "unsupported_survival_stratum",
                "Prediction requires one explicit stratum label column.",
            )
        strata = strata[0] if strata else None
    adapter.stratum = strata
    adapter.sql = None
    if target != "relative_hazard":
        metadata = baseline.metadata.get("analysis", {}) if isinstance(baseline, Dataset) else {}
        if (
            metadata.get("target") != "full_baseline"
            or not metadata.get("full_step_function")
            or metadata.get("model_id") != result.id
            or metadata.get("terms") != list(state.terms)
            or metadata.get("parameters") != state.beta.tolist()
            or metadata.get("source_hash") != result.provenance.get("data_hash")
        ):
            fail(
                "prediction_state_missing",
                "Supply the immutable full baseline from cox_baseline for this saved result.",
            )
        from openecon.resources import plan_workspace

        plan_workspace(
            "saved Cox baseline lookup",
            {
                "SQLite_cache": 2 * 1024**2,
                "baseline_batch_and_gradients": 4096 * 256 * (len(state.terms) + 8),
            },
        )
        owned = stack.enter_context(Spill())
        adapter.sql = owned
        owned.execute(
            "CREATE TABLE baseline(s TEXT,t REAL,h REAL,g BLOB,jump REAL,jg BLOB,PRIMARY KEY(s,t)) WITHOUT ROWID"
        )
        for block in baseline.iter_batches(batch_rows=4096):
            records = []
            for i, (_, row) in enumerate(block.iterrows()):
                label = block["stratum"].iloc[i]
                label = label.item() if hasattr(label, "item") else label
                label = None if pd.isna(label) else label
                key = _stratum_key(label)
                g = torch.tensor(
                    [row[f"gradient[{i}]"] for i in range(len(state.terms))], dtype=torch.float64
                )
                jg = torch.tensor(
                    [row[f"jump_gradient[{i}]"] for i in range(len(state.terms))],
                    dtype=torch.float64,
                )
                records.append(
                    (
                        key,
                        row.time,
                        row.cumulative_hazard,
                        g.numpy().tobytes(),
                        row.hazard_jump,
                        jg.numpy().tobytes(),
                    )
                )
            owned.executemany("INSERT INTO baseline VALUES (?,?,?,?,?,?)", records)
    else:
        # No baseline is needed for a relative hazard; preserve the same encoder
        # while avoiding a disk/source dependency for that identified target.
        adapter.encode = lambda frame, design: Response.encode(adapter, frame, design)
    return _Model(
        result,
        state,
        None,
        "explicit",
        list(result.spec.predictors),
        categories,
        {},
        {},
        set(),
        result.spec.columns.get("offset"),
        None,
        None,
        None,
        None,
        None,
        adapter,
    )


def survival_predict(
    result,
    data,
    *,
    target,
    time=None,
    quantile=0.5,
    baseline=None,
    interval=None,
    alpha=None,
    batch_rows=None,
):
    """Explicit parametric survival response with full ancillary delta covariance.

    Times are measured from zero with constant supplied covariates. Entry-
    conditional and subject-path targets require a different explicit path API.
    Dataset output is indexed, bounded and owned. Never refits the model.
    """
    from openecon.econometrics.postest.prediction import _prediction_frame
    from openecon.econometrics.postest.streaming_prediction import predict_dataset

    if interval not in {None, "mean"}:
        fail(
            "unsupported_prediction_interval",
            "Only saved-parameter delta intervals are implemented.",
        )
    if (
        isinstance(result, ResultBundle)
        and result.spec.estimator == "stcox"
        and target == "quantile"
        and interval is not None
    ):
        fail(
            "unsupported_prediction_interval",
            "Cox step quantiles do not have a regular parameter delta interval.",
        )
    with torch.device("cpu"), torch.inference_mode(False), ExitStack() as stack:
        model = (
            _cox_model(result, target, time, quantile, baseline, stack)
            if isinstance(result, ResultBundle) and result.spec.estimator == "stcox"
            else _parametric_model(result, target, time, quantile)
        )
        significance = model.state.alpha if alpha is None else _alpha(alpha)
        if isinstance(data, Dataset):
            return predict_dataset(
                model,
                data,
                kind="response",
                alpha=significance,
                interval=interval,
                term=None,
                outcome=target,
                batch_rows=batch_rows,
            )
        if batch_rows is not None:
            fail("invalid_batch_size", "batch_rows applies to Dataset data.")
        return _prediction_frame(
            model,
            data,
            kind="response",
            significance=significance,
            interval=interval,
            term=None,
            outcome=target,
        )
