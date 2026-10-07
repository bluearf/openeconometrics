"""Full-source VAR: disk-ordered lag replay, global system factors and tests."""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.resources import plan_workspace
from openecon.streaming_design import numeric_values
from .core import kernel_call, make_spec, wald_test
from .ordered_replay import OrderedReplay
from .streaming_linear import _Notes, _factor, _result
from .streaming_systems import fit_streaming_systems
from .var import impulse, kernels
from .var.common import system_names
from .var.estimators import MAX_STORED_MOMENT, MAX_STORED_RESPONSES, _wald_rows, check_model_size


def _options(spec, name):
    return _Notes(spec).option(name)


class _LagSource:
    def __init__(self, ordered, p, trend, constant):
        self.ordered, self.p, self.trend, self.constant = ordered, p, trend, constant
        self.spec = ordered.spec
        self.names = system_names(self.spec)
        self.categories = {}
        self.exogenous = []
        for name in self.spec.predictors:
            if name in {"Intercept", "trend"} and (constant or trend):
                raise AnalysisError("duplicate_terms", "Exogenous names collide with VAR deterministic terms.")
            if name in self.spec.categorical:
                dtype = ordered.dtypes[name]
                if isinstance(dtype, pd.CategoricalDtype):
                    levels = list(dtype.categories)
                else:
                    index = ordered.columns.index(name)
                    levels = [row[0] for row in ordered.db.execute(f"SELECT DISTINCT c{index} FROM rows WHERE c{index} IS NOT NULL LIMIT 385")]
                    try:
                        levels.sort()
                    except TypeError as exc:
                        raise AnalysisError("ambiguous_categories", "Mixed labels need a declared categorical dtype.") from exc
                if not 2 <= len(levels) <= 384:
                    raise AnalysisError("model_too_wide", "VAR categorical expansion exceeds its bounded width or has only one level.")
                self.categories[name] = {"levels": levels, "reference": levels[0], "coding": "treatment_drop_first"}
                self.exogenous.extend(f"{name}[{value}]" for value in levels[1:])
            else:
                self.exogenous.append(name)
        self.regressors = [f"L{j}.{name}" for name in self.names for j in range(1, p+1)]+self.exogenous+(["trend"] if trend else [])+(["Intercept"] if constant else [])
        if len(self.regressors) != len(set(self.regressors)) or set(self.regressors)&set(self.names):
            raise AnalysisError("duplicate_terms", "Derived lag/exogenous names collide; rename the conflicting column.")
        self.width = len(self.regressors)
        check_model_size(self.width*len(self.names))
        if self.width+len(self.names) > 384:
            raise AnalysisError("model_too_wide", "Combined VAR lag/outcome factor exceeds384 columns.")
        self.plan = plan_workspace("bounded VAR lag/factor/covariance buffers", {
            "lag_and_score_rows": 96*ordered.rows*(self.width+len(self.names)),
            "lag_history": 128*(max(p, int(_options(self.spec, "lm_lags")))+1)*max(self.width, 1),
            "system_factors_and_parameter_covariance": 128*(self.width*len(self.names))**2,
            "QR_tree": 512*(self.width+len(self.names))**2,
            "sorted_source_cache": 8*1024**2,
        }, budget_bytes=ordered.budget)
        self.last_levels = None

    def matrix(self, frame, terms):
        return torch.stack([torch.ones(len(frame), dtype=torch.float64) if term == "Intercept"
                            else numeric_values(frame[term], term) for term in terms], dim=1) if terms else torch.empty((len(frame), 0), dtype=torch.float64)

    def factory(self):
        carry, seen = None, 0
        for block in self.ordered.source.iter_batches(batch_rows=self.ordered.rows):
            old = 0 if carry is None else len(carry)
            joined = pd.concat([carry, block], ignore_index=True) if carry is not None else block.reset_index(drop=True)
            first = max(self.p, old)
            output = joined.iloc[first:][self.names].copy().reset_index(drop=True)
            for name in self.names:
                for lag in range(1, self.p+1):
                    output[f"L{lag}.{name}"] = joined[name].iloc[first-lag:len(joined)-lag].to_numpy()
            for name in self.spec.predictors:
                values = joined[name].iloc[first:]
                if name in self.categories:
                    for level in self.categories[name]["levels"][1:]:
                        output[f"{name}[{level}]"] = (values == level).to_numpy(dtype="float64")
                else:
                    output[name] = values.to_numpy()
            if self.trend:
                output["trend"] = list(range(seen-old+first+1, seen+len(block)+1))
            if self.constant:
                output["Intercept"] = 1.
            seen += len(block)
            carry = joined.tail(self.p).copy() if self.p else None
            self.last_levels = joined[self.names].tail(self.p).copy() if self.p else joined[self.names].iloc[:0]
            if len(output):
                yield output

    def dataset(self):
        return Dataset.from_batches(self.factory, [*self.names, *self.regressors], row_count=self.ordered.count-self.p)


def _fit(ordered):
    spec = ordered.spec
    names = system_names(spec)
    p, k = int(_options(spec, "lags")), len(names)
    t = ordered.count-p
    constant = bool(spec.intercept) and bool(_options(spec, "constant"))
    trend, small, dfk = (bool(_options(spec, option)) for option in ("trend", "small", "dfk"))
    lm_lags = int(_options(spec, "lm_lags"))
    maxlag = p if _options(spec, "maxlag") is None else int(_options(spec, "maxlag"))
    lagged = _LagSource(ordered, p, trend, constant)
    if t <= lagged.width:
        raise AnalysisError("insufficient_observations", "VAR needs more usable periods than regressors.")
    # Screen exogenous regressors against deterministic terms before lagged
    # endogenous columns enter, exactly like the dense VAR contract.
    deterministic = (["Intercept"] if constant else [])+(["trend"] if trend else [])
    exo_terms = deterministic+lagged.exogenous
    if exo_terms:
        def exogenous_blocks():
            for frame in lagged.factory():
                x = lagged.matrix(frame, exo_terms)
                yield x, torch.zeros(len(frame), dtype=torch.float64), torch.ones(len(frame), dtype=torch.float64)
        factor, _ = _factor(exogenous_blocks, len(exo_terms))
        kept, omitted = collinear_columns(factor[:, :len(exo_terms)])
        omitted_terms = [exo_terms[index] for index in omitted]
        lagged.exogenous = [term for term in lagged.exogenous if term not in omitted_terms]
    else:
        omitted_terms = []
    terms = [f"L{j}.{name}" for name in names for j in range(1, p+1)]+lagged.exogenous+(["trend"] if trend else [])+(["Intercept"] if constant else [])
    predictors = [term for term in terms if term != "Intercept"]
    equations = [dict(y=name, x=predictors, constant=constant) for name in names]
    system_spec = make_spec("sureg", outcome=names[0], intercept=constant, alpha=spec.alpha,
                            columns={"system": list(dict.fromkeys([*names, *predictors]))},
                            options={"equations": equations, "small": small, "dfk": small or dfk})
    base = fit_streaming_systems(system_spec, lagged.dataset())
    all_omitted = base.provenance.get("omitted_terms", [])
    if all_omitted:
        raise AnalysisError("collinear_system", "Lagged endogenous variables are dependent on the other regressors; remove the redundant variable or lag.")
    source_terms = base.extra["equations"][0]["terms"]
    terms = [term for term in terms if term in source_terms]
    m, df = len(terms), t-len(terms)
    permutation = [j*m+source_terms.index(term) for j in range(k) for term in terms]
    parameters = torch.tensor([base.coefficients[index].estimate for index in permutation], dtype=torch.float64)
    covariance = torch.tensor(base.covariance_matrix, dtype=torch.float64)[permutation][:, permutation]
    coefficients = parameters.reshape(k, m)
    sigma = torch.tensor(base.extra["sigma"], dtype=torch.float64)
    sigma_ml = sigma*(df/t) if small or dfk else sigma
    actual_sigma = sigma_ml*(t/df) if dfk else sigma_ml
    inverse = torch.tensor(base.covariance_matrix, dtype=torch.float64)[:m, :m]/sigma[0, 0]
    order = [source_terms.index(term) for term in terms]
    inverse = inverse[order][:, order]
    diagnostics = _diagnostics(lagged, terms, coefficients, inverse, actual_sigma, sigma_ml, lm_lags)
    if spec.covariance == "robust":
        meat = _CompensatedSum((k*m, k*m))
        for frame in lagged.factory():
            x = lagged.matrix(frame, terms)
            y = torch.stack([numeric_values(frame[name], name) for name in names], dim=1)
            u = y-x@coefficients.T
            for i in range(k):
                for j in range(i, k):
                    block = x.T@(x*(u[:, i]*u[:, j])[:, None])
                    matrix = torch.zeros((k*m, k*m), dtype=torch.float64)
                    matrix[i*m:(i+1)*m, j*m:(j+1)*m] = block
                    if i != j:
                        matrix[j*m:(j+1)*m, i*m:(i+1)*m] = block.T
                    meat.add(matrix)
        bread = torch.kron(torch.eye(k, dtype=torch.float64), inverse)
        covariance = bread@meat.value@bread*(t/df if small or dfk else t/(t-1))
    ll = kernels.log_likelihood(kernel_call(kernels.logdet, kernel_call(kernels.spd_factor, sigma_ml)), t, k)
    logdet = kernel_call(kernels.logdet, kernel_call(kernels.spd_factor, sigma_ml))
    parameters_count = k*m
    metrics = {"log_likelihood": ll, "aic": -2*ll+2*parameters_count,
               "bic": -2*ll+math.log(t)*parameters_count,
               "hqic": -2*ll+2*math.log(math.log(t))*parameters_count,
               "aic_per_obs": (-2*ll+2*parameters_count)/t,
               "hqic_per_obs": (-2*ll+2*math.log(math.log(t))*parameters_count)/t,
               "sbic_per_obs": (-2*ll+math.log(t)*parameters_count)/t,
               "fpe": math.exp(logdet+k*(math.log(t+m)-math.log(t-m))),
               "det_sigma_ml": math.exp(logdet), "n_equations": k, "n_lags": p,
               "df_eq": m, "df_model": parameters_count, "df_resid": df, "T": t}
    tests, extra, equations = _extras(spec, lagged, terms, coefficients, covariance, actual_sigma, sigma_ml, inverse, diagnostics)
    tests.update(diagnostics["tests"])
    extra.update(diagnostics["extra"])
    extra["equations"] = equations
    try:
        extra["lag_order_selection"] = _lag_order(ordered, names, maxlag, constant, trend, spec.alpha, lagged.exogenous)
    except AnalysisError as exc:
        if exc.code not in {"insufficient_observations", "singular_residual_covariance", "perfect_fit"}:
            raise
    positions = [row[0] for row in ordered.db.execute("SELECT pos FROM rows WHERE valid=1 ORDER BY t,pos LIMIT 400 OFFSET ?", (p,))]
    reporting = pd.concat(list(_head(lagged.factory(), 400)), ignore_index=True)
    facade = _Facade(spec, ordered, reporting, positions, t, lagged.categories, base.provenance)
    notes = _Notes(spec)
    if omitted_terms:
        notes.warn("Omitted collinear exogenous terms: "+", ".join(omitted_terms))
    if not extra["stability"]["stable"]:
        notes.warn("The VAR does not satisfy the stability condition.")
    predictions = []
    for record, position in zip(base.predictions, positions, strict=False):
        predictions.append({**record, "row": position})
    result = _result(facade, terms=[f"{name}:{term}" for name in names for term in terms],
                   beta=parameters, covariance=covariance,
                   info={"covariance": spec.covariance, "df_resid": df, "df_inference": df if small else None,
                         "small": small, "dfk": dfk, "correction": "global VAR system covariance"},
                   metrics=metrics, notes=notes, predictions=predictions, tests=tests,
                   solver="global_tsqr_disk_ordered_var", diagnostics={"equations": k, "regressors_per_equation": m},
                   extra=extra, resource=lagged.plan.record(), title="Vector autoregression", use_t=small)
    for name, rows in zip(names, [result.coefficients[i*m:(i+1)*m] for i in range(k)], strict=True):
        for row in rows:
            row.equation = name
    return result


def _diagnostics(lagged, terms, coefficients, inverse, sigma, sigma_ml, lm_lags):
    k, m = coefficients.shape
    t = lagged.ordered.count-lagged.p
    chol = kernel_call(kernels.spd_factor, sigma)
    third, fourth = _CompensatedSum((k,)), _CompensatedSum((k,))
    squares = _CompensatedSum((k,))
    moment = _WeightedMoments(m, intercept=lagged.constant)
    order = ([terms.index("Intercept")]+[i for i, term in enumerate(terms) if term != "Intercept"]
             if lagged.constant else list(range(m)))
    for frame in lagged.factory():
        x = lagged.matrix(frame, terms)
        y = torch.stack([numeric_values(frame[name], name) for name in lagged.names], dim=1)
        residual = y-x@coefficients.T
        white = torch.linalg.solve_triangular(chol, residual.T, upper=False).T
        third.add(white.pow(3).sum(0))
        fourth.add(white.pow(4).sum(0))
        squares.add(residual.square().sum(0))
        moment.add(x[:, order], torch.ones(len(x), dtype=torch.float64), False)
    means = moment.anchor+moment.magnitude*moment.mean
    scales = moment.magnitude*(moment.m2.value/moment.mass).clamp_min(0).sqrt()
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    if lagged.constant:
        means[0], scales[0] = 0., 1.
    def normalized(frame):
        x = lagged.matrix(frame, terms)[:, order]
        return ((x-means)/scales if lagged.constant else x/scales)
    def factor_blocks():
        for frame in lagged.factory():
            x = normalized(frame)
            yield x, torch.zeros(len(x), dtype=torch.float64), torch.ones(len(x), dtype=torch.float64)
    factor, _ = _factor(factor_blocks, m)
    r = factor[:m, :m]
    ri = torch.linalg.solve_triangular(r, torch.eye(m, dtype=torch.float64), upper=True)
    bread = ri@ri.T
    cross = [_CompensatedSum((m, k)) for _ in range(lm_lags)]
    inner = [_CompensatedSum((k, k)) for _ in range(lm_lags)]
    residual_moment = [_CompensatedSum((k, k)) for _ in range(lm_lags)]
    tail = torch.empty((0, k), dtype=torch.float64)
    for frame in lagged.factory():
        raw = lagged.matrix(frame, terms)
        y = torch.stack([numeric_values(frame[name], name) for name in lagged.names], dim=1)
        u = y-raw@coefficients.T
        x = normalized(frame)
        combined = torch.cat((tail, u))
        for lag in range(1, lm_lags+1):
            ids = len(tail)+torch.arange(len(u))-lag
            valid = ids >= 0
            earlier = torch.zeros_like(u)
            earlier[valid] = combined[ids[valid]]
            cross[lag-1].add(x.T@earlier)
            inner[lag-1].add(earlier.T@earlier)
            residual_moment[lag-1].add(earlier.T@u)
        tail = combined[-lm_lags:].clone() if lm_lags else combined[:0]
    b1, b2 = third.value/t, fourth.value/t
    skew, kurt = t*b1.square()/6, t*(b2-3).square()/24
    rows = []
    for i, name in enumerate([*lagged.names, "ALL"]):
        if i < k:
            s, q, df = float(skew[i]), float(kurt[i]), 1
            row = {"equation": name, "skewness": float(b1[i]), "kurtosis": float(b2[i])}
        else:
            s, q, df = float(skew.sum()), float(kurt.sum()), k
            row = {"equation": name, "skewness": None, "kurtosis": None}
        rows.append({**row, "skewness_chi2": s, "skewness_p": chi2_sf(s, df),
                     "kurtosis_chi2": q, "kurtosis_p": chi2_sf(q, df), "jarque_bera": s+q,
                     "jarque_bera_df": 2*df, "jarque_bera_p": chi2_sf(s+q, 2*df)})
    tests = {"normality": {"statistic": rows[-1]["jarque_bera"], "df": 2*k,
                           "p_value": rows[-1]["jarque_bera_p"], "distribution": "chi2",
                           "label": "Jarque-Bera test that the disturbances are jointly normal"}}
    logdet = kernel_call(kernels.logdet, kernel_call(kernels.spd_factor, sigma_ml))
    for lag in range(1, lm_lags+1):
        if lag >= t or t-m-k-.5 <= 0:
            continue
        v = inner[lag-1].value-cross[lag-1].value.T@bread@cross[lag-1].value
        e = residual_moment[lag-1].value
        try:
            explained = e.T@torch.cholesky_solve(e, kernel_call(kernels.spd_factor, v))
            augmented = kernel_call(kernels.logdet, kernel_call(kernels.spd_factor, sigma_ml-explained/t))
        except AnalysisError:
            continue
        value = max(0., (t-m-k-.5)*(logdet-augmented))
        tests[f"lm_autocorrelation_L{lag}"] = {"statistic": value, "df": k*k,
                                              "p_value": chi2_sf(value, k*k), "distribution": "chi2",
                                              "label": f"LM test of no residual autocorrelation at lag {lag}"}
    inverse_order = [order.index(i) for i in range(m)]
    gram = (r.T@r/t)*scales[:, None]*scales[None, :]
    if lagged.constant:
        gram[0] = 0.
        gram[:, 0] = 0.
        gram[0, 0] = 1.
    return {"tests": tests, "extra": {"normality": rows}, "rss": squares.value,
            "moment": gram[inverse_order][:, inverse_order],
            "means": means[1:] if lagged.constant else None}


def _extras(spec, lagged, terms, coefficients, covariance, sigma, sigma_ml, inverse, diagnostics):
    k, m = coefficients.shape
    p, n, t = lagged.p, lagged.ordered.count, lagged.ordered.count-lagged.p
    small, dfk = bool(_options(spec, "small")), bool(_options(spec, "dfk"))
    reference = t-m if small else None
    params = coefficients.reshape(-1)
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    equations = []
    # Stable centered outcome TSS, instead of subtracting two huge raw sums.
    moments = _WeightedMoments(k+1, intercept=True)
    for frame in lagged.factory():
        y = torch.stack([torch.ones(len(frame), dtype=torch.float64), *[numeric_values(frame[name], name) for name in lagged.names]], dim=1)
        moments.add(y, torch.ones(len(y), dtype=torch.float64), False)
    total = moments.m2.value[1:]*moments.magnitude[1:].square()
    if not lagged.constant:
        total += moments.mass*(moments.anchor[1:]+moments.magnitude[1:]*moments.mean[1:]).square()
    for i, name in enumerate(lagged.names):
        record = {"equation": name, "parms": m, "rmse": math.sqrt(float(diagnostics["rss"][i])/(t-m)),
                  "r_squared": 1-float(diagnostics["rss"][i]/total[i]) if float(total[i]) > 0 else None}
        if slopes:
            test = wald_test(params, covariance, [i*m+c for c in slopes], df_resid=reference)
            record.update({key: test[key] for key in ("statistic", "df", "df2", "p_value", "distribution") if key in test})
        equations.append(record)
    granger_groups, exclusion_groups = [], []
    for i, name in enumerate(lagged.names):
        others = [j for j in range(k) if j != i]
        for other in others:
            granger_groups.append(({"equation": name, "excluded": lagged.names[other]}, [i*m+other*p+j for j in range(p)]))
        if others and p:
            granger_groups.append(({"equation": name, "excluded": "ALL"}, [i*m+other*p+j for other in others for j in range(p)]))
    for j in range(p):
        for i, name in enumerate(lagged.names):
            exclusion_groups.append(({"lag": j+1, "equation": name}, [i*m+other*p+j for other in range(k)]))
        exclusion_groups.append(({"lag": j+1, "equation": "ALL"}, [i*m+other*p+j for i in range(k) for other in range(k)]))
    granger = _wald_rows(params, covariance, granger_groups, reference) if p else []
    exclusion = _wald_rows(params, covariance, exclusion_groups, reference)
    tests = {}
    for row in granger:
        if row["excluded"] == "ALL":
            tests[f"granger_all_{row['equation']}"] = {key: value for key, value in row.items() if key not in {"equation", "excluded"}}
    for row in exclusion:
        if row["equation"] == "ALL":
            tests[f"lag_exclusion_L{row['lag']}"] = {key: value for key, value in row.items() if key not in {"equation", "lag"}}
    a = kernels.lag_matrices(coefficients, k, p)
    roots = kernel_call(kernels.companion_eigenvalues, a)
    stable = all(root["modulus"] < 1 for root in roots)
    extra = {"layout": {"variables": lagged.names, "lags": p, "exogenous": lagged.exogenous,
                         "trend": lagged.trend, "constant": lagged.constant, "n_regressors": m,
                         "regressors": terms, "small": small, "dfk": dfk},
             "sigma": sigma, "sigma_ml": sigma_ml, "stability": {"eigenvalues": roots, "stable": stable,
                 "convention": "eigenvalues of the companion matrix; stable when every modulus is below1"},
             "granger": granger, "lag_exclusion": exclusion,
             "forecast": {"last_values": torch.tensor(lagged.last_levels.to_numpy(), dtype=torch.float64),
                          "last_period": lagged.ordered.last_period, "last_trend": n,
                          "moment": diagnostics["moment"] if m <= MAX_STORED_MOMENT else None,
                          "means": diagnostics["means"]}}
    steps = int(_options(spec, "irf_steps"))
    kinds = _options(spec, "irf_kinds")
    kinds = list(impulse.KINDS) if kinds is None else list(dict.fromkeys(kinds))
    if any(kind not in impulse.KINDS for kind in kinds):
        raise AnalysisError("invalid_option", "Unknown impulse-response kind.")
    if p and (steps+1)*k*k <= MAX_STORED_RESPONSES:
        positions = impulse.alpha_positions(k, p, m)
        responses = kernel_call(impulse.impulse_responses, a, sigma, steps,
                                cov_alpha=covariance[positions][:, positions], n_obs=t)
        record = {"steps": steps, "variables": lagged.names, "kinds": kinds,
                  "convention": "array[step][response][impulse]; Cholesky variable order"}
        for kind in kinds:
            record[kind] = getattr(responses, kind)
        if "orthogonalized" in kinds:
            record["fevd"] = responses.fevd
        record["se"] = {name: value for name, value in responses.se.items()
                        if name in kinds or name == "fevd" and "orthogonalized" in kinds} if responses.se is not None else None
        extra["irf"] = record
    return tests, extra, equations


def _lag_order(ordered, names, maxlag, constant, trend, alpha, exogenous):
    lagged = _LagSource(ordered, maxlag, trend, constant)
    t, k = ordered.count-maxlag, len(names)
    d0 = len(exogenous)+int(constant)+int(trend)
    if t < d0+k*maxlag+k+1:
        raise AnalysisError("insufficient_observations", "The longest lag leaves too few periods for lag-order statistics.")
    rows, previous = [], None
    source = lagged.dataset()
    for lag in range(maxlag+1):
        regressors = [*exogenous, *(["trend"] if trend else []), *[f"L{j}.{name}" for j in range(1, lag+1) for name in names]]
        options = {"equations": [dict(y=name, x=regressors, constant=constant) for name in names]}
        specification = make_spec("sureg", outcome=names[0], intercept=constant,
                                  columns={"system": [*names, *regressors]}, options=options)
        fit = fit_streaming_systems(specification, source)
        sigma = torch.tensor(fit.extra["sigma"], dtype=torch.float64)
        logdet = kernel_call(kernels.logdet, kernel_call(kernels.spd_factor, sigma))
        ll = kernels.log_likelihood(logdet, t, k)
        m, parameters = d0+k*lag, k*(d0+k*lag)
        lr = max(0., 2*(ll-previous)) if previous is not None else None
        rows.append({"lag": lag, "ll": ll, "lr": lr, "df": k*k if lr is not None else None,
                     "p_value": chi2_sf(lr, k*k) if lr is not None else None,
                     "fpe": math.exp(logdet+k*(math.log(t+m)-math.log(t-m))) if t > m else None,
                     "aic": (-2*ll+2*parameters)/t, "hqic": (-2*ll+2*math.log(math.log(t))*parameters)/t,
                     "sbic": (-2*ll+math.log(t)*parameters)/t})
        previous = ll
    selected = {name: min((row[name], row["lag"]) for row in rows if row[name] is not None)[1]
                for name in ("fpe", "aic", "hqic", "sbic")}
    rejected = [row["lag"] for row in rows[1:] if row["p_value"] < alpha]
    selected["lr"] = max(rejected) if rejected else 0
    return {"maxlag": maxlag, "n_obs": t, "rows": rows, "selected": selected, "alpha": alpha,
            "convention": "all candidate lags on the same longest-lag sample"}


def _head(iterator, limit):
    used = 0
    try:
        for frame in iterator:
            take = min(limit-used, len(frame))
            if take:
                yield frame.iloc[:take].copy()
                used += take
            if used == limit:
                break
    finally:
        iterator.close()


class _Facade(SimpleNamespace):
    def __init__(self, spec, ordered, sample, positions, nobs, categories, provenance):
        super().__init__(spec=spec, sample=sample, sample_positions=positions, nobs=nobs,
                         nrows=nobs, original_count=ordered.original_count,
                         categories=categories, notes={"omitted_terms": []})
        self.record = {**provenance, **ordered.provenance()}
        digest = hashlib.sha256()
        for records in _position_batches(ordered, ordered.count-nobs):
            digest.update(torch.tensor(records, dtype=torch.int64).numpy().astype("<i8", copy=False).tobytes())
        self.record.update({"data_hash": ordered.raw_hash, "sample_positions_hash": digest.hexdigest(),
                            "sample_hash": hashlib.sha256((ordered.raw_hash+digest.hexdigest()).encode()).hexdigest(),
                            "prediction_sample": "first400 retained periods in time order",
                            "sample_position_count": nobs})

    def provenance(self):
        return dict(self.record)


def _position_batches(ordered, offset):
    cursor = ordered.db.execute("SELECT pos FROM rows WHERE valid=1 ORDER BY t,pos LIMIT -1 OFFSET ?", (offset,))
    while records := cursor.fetchmany(65536):
        yield [row[0] for row in records]


def fit_streaming_var(spec, source):
    with OrderedReplay(spec, source) as ordered:
        result = kernel_call(_fit, ordered)
        ordered.verify_original()
        return result
