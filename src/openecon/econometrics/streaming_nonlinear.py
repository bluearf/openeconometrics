"""Full-source nonlinear LS and robust regression on bounded Dataset replays.

NL accumulates one global objective/Jacobian system at each iteration. Rreg
uses full-source Cook screening and exact SQLite-backed global medians/MADs;
it never substitutes per-block regressions or per-block order statistics.
Methods follow https://www.stata.com/manuals/rnl.pdf and
https://www.stata.com/manuals15/rrreg.pdf. Native arithmetic is Torch float64;
SQLite is owned scratch storage, not a numerical fitting dependency.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import hc_residuals
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.models import ModelSpec, ResultBundle
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import kernel_call, wald_test
from .quantile import formula as formulas
from .quantile import nl as nonlinear
from .quantile import rreg as robust
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _factor, _finite, _result, _solve, _weights

SUPPORTED = frozenset({"nl", "rreg"})


@dataclass
class _NLStats:
    rss: float
    gram: Tensor | None
    gradient: Tensor | None
    low: Tensor | None
    high: Tensor | None
    centered_tss: float
    outcome_square_sum: float


def _formula_values(sample, batch, formula, theta, jacobian=True):
    columns = {name: batch.numeric(name) for name in formula.columns}
    return formulas.evaluate(formula, columns, theta, len(batch.frame), jacobian=jacobian)


def _nl_stats(sample, formula, theta, *, jacobian=True):
    k = len(theta)
    rss, yy = _CompensatedSum(()), _CompensatedSum(())
    gram = _CompensatedSum((k, k)) if jacobian else None
    gradient = _CompensatedSum((k,)) if jacobian else None
    low, high = (torch.full((k,), math.inf, dtype=torch.float64), torch.full((k,), -math.inf, dtype=torch.float64)) if jacobian else (None, None)
    moments = _WeightedMoments(2, intercept=True)
    valid = True
    for batch in sample.batches():
        y, weights = batch.numeric(sample.spec.outcome), _weights(sample, batch)
        fitted, j = _formula_values(sample, batch, formula, theta, jacobian)
        if not bool(torch.isfinite(fitted).all()) or (jacobian and not bool(torch.isfinite(j).all())):
            valid = False
            continue  # Drain every pass even for a rejected trial.
        resid = y-fitted
        _finite(resid, weights*resid.square(), weights*y.square())
        rss.add((weights*resid.square()).sum())
        yy.add((weights*y.square()).sum())
        moments.add(torch.stack((torch.ones_like(y), y), 1), weights, False)
        if jacobian:
            weighted = j*weights[:, None]
            g, v = weighted.T@j, weighted.T@resid
            _finite(g, v)
            gram.add(g)
            gradient.add(v)
            low, high = torch.minimum(low, j.amin(0)), torch.maximum(high, j.amax(0))
    if not valid:
        return _NLStats(math.inf, None, None, None, None, math.nan, math.nan)
    tss = float(moments.m2.value[1])*float(moments.magnitude[1])**2*moments.weight_max
    if not math.isfinite(tss):
        raise AnalysisError("numerical_failure", "The full-source total sum of squares exceeds float64 precision; rescale the outcome.")
    return _NLStats(float(rss.value), gram.value if jacobian else None,
                    gradient.value if jacobian else None, low, high, tss, float(yy.value))


def _nl_optimize(sample, formula, start, tolerance, limit):
    theta = start.clone()
    current = _nl_stats(sample, formula, theta)
    if not math.isfinite(current.rss):
        raise AnalysisError("invalid_start", "The nonlinear function or its derivatives are not finite at the starting values.")
    floor = nonlinear._EXACT_FIT*current.outcome_square_sum
    evaluations, damping, history = 1, 0., [current.rss]
    predicted = math.inf
    for iteration in range(1, limit+1):
        accepted = None
        while accepted is None:
            step = nonlinear._step(current.gram, current.gradient, damping)
            if step is None:
                damping = max(10*damping, 1e-6)
            else:
                if damping <= nonlinear._STATIONARY_DAMPING:
                    predicted = float(current.gradient@step)
                    if predicted <= tolerance**2*current.rss+floor:
                        return theta, current, iteration-1, evaluations, damping, predicted/max(current.rss, 1e-300), history
                for halving in range(nonlinear._HALVINGS+1 if damping == 0 else 1):
                    trial = theta+step*.5**halving
                    trial_stats = _nl_stats(sample, formula, trial, jacobian=False)
                    evaluations += 1
                    if math.isfinite(trial_stats.rss) and trial_stats.rss <= current.rss:
                        accepted = trial, step*.5**halving, trial_stats.rss
                        break
                if accepted is None:
                    damping = max(10*damping, 1e-4)
            if accepted is None and damping > nonlinear._MAX_DAMPING:
                raise AnalysisError("nonconvergence", "No global nonlinear LS step reduces the objective; revise starting values or identification.")
        theta, step, trial_rss = accepted
        small = bool((step.abs() <= tolerance*(theta.abs()+nonlinear._PARAMETER_FLOOR)).all())
        settled = current.rss-trial_rss <= tolerance*trial_rss+floor
        previous = current.rss
        current = _nl_stats(sample, formula, theta)
        evaluations += 1
        if not math.isfinite(current.rss):
            raise AnalysisError("numerical_failure", "The nonlinear derivatives are not finite at the accepted solution.")
        history.append(current.rss)
        del history[:-50]
        if small and settled and damping <= 1:
            return theta, current, iteration, evaluations, damping, max(previous-current.rss, 0.)/max(current.rss, 1e-300), history
        damping = damping/10 if damping > 1e-10 else 0.
    raise AnalysisError("nonconvergence", f"Nonlinear least squares did not converge in {limit} global iterations.")


def _nl_covariance(sample, formula, theta, bread, rss):
    k, n = len(theta), sample.nobs
    df = n-k
    kind = "HC1" if sample.spec.covariance == "robust" else sample.spec.covariance
    info = {"covariance": sample.spec.covariance, "df_resid": df, "df_inference": df}
    meat, predictions = _CompensatedSum((k, k)), []
    columns = registry.cluster_columns(sample.spec)
    accumulator = ClusterAccumulator(k) if kind == "cluster" else None
    try:
        for batch in sample.batches():
            y, w = batch.numeric(sample.spec.outcome), _weights(sample, batch)
            fitted, j = _formula_values(sample, batch, formula, theta)
            resid = y-fitted
            _finite(fitted, j, resid)
            if kind == "cluster":
                accumulator.add(encode_cluster_labels(batch.frame[columns[0]]), j*(resid*w)[:, None])
            elif kind != "nonrobust":
                leverage = ((j@bread)*j).sum(1) if kind in {"HC2", "HC3"} else None
                if leverage is not None and sample.spec.weight_type != "fweight":
                    leverage *= w
                adjusted = kernel_call(hc_residuals, resid, leverage, kind)
                score = j*(adjusted*(w.sqrt() if sample.spec.weight_type == "fweight" else w))[:, None]
                _finite(score)
                meat.add(score.T@score)
            take = min(400-len(predictions), len(y))
            for row, observed, predicted in zip(batch.positions[:take].tolist(), y[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row": row, "observed": observed, "fitted": predicted, "residual": observed-predicted})
        if kind == "nonrobust":
            covariance = bread*(rss/df)
            info["correction"] = "classical: SSR/(N-K)"
        else:
            matrix = meat.value
            factor = n/df if kind == "HC1" else 1.
            if kind == "cluster":
                matrix, groups = accumulator.finish()
                factor = groups/(groups-1)*(n-1)/df
                info.update({"cluster_count": groups, "cluster_columns": columns,
                             "cluster_column": columns[0], "cluster_df": groups-1,
                             "df_inference": groups-1, "cluster_spill": accumulator.diagnostics})
            info.update({"small_sample_correction": factor,
                         "correction": {"HC1":"HC1: N/(N-K)", "HC2":"HC2: residual / sqrt(1-h)",
                                        "HC3":"HC3: residual / (1-h)", "cluster":"CR1: G/(G-1) * (N-1)/(N-K)"}[kind]})
            covariance = bread@(matrix*factor)@bread
        _finite(covariance)
        return (covariance+covariance.T)/2, info, predictions
    finally:
        if accumulator is not None:
            accumulator.close()


def _fit_nl(spec, source, batch_rows):
    formula = formulas.parse(spec.options.get("formula"))
    if spec.outcome in formula.columns:
        raise AnalysisError("invalid_formula", "The outcome must not appear in the nonlinear regression function.")
    if set(spec.predictors)-set(formula.columns):
        raise AnalysisError("invalid_spec", "Predictors must be columns used in the nonlinear formula.")
    projected = spec.model_copy(update={"predictors":list(formula.columns)})
    sample, notes = ReplaySample(projected, source, batch_rows=batch_rows), _Notes(spec)
    sample.prepare()
    k = len(formula.parameters)
    if k > 384:
        raise AnalysisError("model_too_wide", "Replay nonlinear LS supports up to 384 parameters.")
    if sample.nrows <= k:
        raise AnalysisError("insufficient_observations", "Nonlinear LS needs more physical observations than parameters.")
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "pweights require robust or cluster covariance.")
    if spec.covariance == "cluster" and len(registry.cluster_columns(spec)) != 1:
        raise AnalysisError("unsupported_cluster_dimensions", "Nonlinear replay covariance supports one cluster dimension.")
    resource = sample.plan_rows("full-source nonlinear LS", {
        "nonlinear_factors": 512*(k+1)**2,
        "nonlinear_cluster_cache": 12*1024**2 if spec.covariance == "cluster" else 0,
    }, 32*len(formula.tape)*(k+2)+128*(k+len(formula.columns)+4)).record()
    start = nonlinear._start_values(notes, formula)
    tolerance, limit = float(notes.option("tolerance")), int(notes.option("max_iterations"))
    if not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tolerance must lie strictly between 0 and 1.")
    theta, stats, iterations, evaluations, damping, predicted, history = _nl_optimize(sample, formula, start, tolerance, limit)
    if stats.rss <= nonlinear._EXACT_FIT*stats.outcome_square_sum:
        raise AnalysisError("perfect_fit", "The nonlinear function fits exactly; standard errors are undefined.")

    def final_jacobian():
        for batch in sample.batches():
            fitted, j = _formula_values(sample, batch, formula, theta)
            yield j, batch.numeric(spec.outcome)-fitted, _weights(sample, batch)

    try:
        factor, tsqr = _factor(final_jacobian, k)
        _, bread, condition = _solve(factor, k)
    except AnalysisError as error:
        if error.code != "singular_design":
            raise
        raise AnalysisError("not_identified", "The final nonlinear Jacobian is rank deficient; remove or fix redundant parameters.") from error
    covariance, info, predictions = _nl_covariance(sample, formula, theta, bread, stats.rss)
    constant = next((name for index, name in enumerate(formula.parameters)
                     if (centre := float(stats.high[index]/2+stats.low[index]/2)) != 0
                     and float(stats.high[index]-stats.low[index]) <= 1e-12*abs(centre)), None)
    tss = stats.centered_tss if constant is not None else stats.outcome_square_sum
    n, df, c = sample.nobs, sample.nobs-k, int(constant is not None)
    r2 = 1-stats.rss/tss if tss > 0 else None
    metrics = {"rss":stats.rss, "tss":tss, "rmse":math.sqrt(stats.rss/df),
               "r_squared":r2, "adjusted_r_squared":None if r2 is None else 1-(1-r2)*(n-c)/df,
               "iterations":iterations, "df_model":k-c, "df_resid":df}
    if spec.weights is None or spec.weight_type == "fweight":
        metrics["log_likelihood"] = -.5*n*(math.log(2*math.pi)+math.log(stats.rss/n)+1)
    result = _result(sample, terms=list(formula.parameters), beta=theta, covariance=covariance,
                     info=info, metrics=metrics, notes=notes, predictions=predictions, tests={},
                     solver="gauss_newton_levenberg_marquardt", resource=resource,
                     diagnostics={**tsqr, "converged":True, "iterations":iterations,
                                  "function_evaluations":evaluations, "final_damping":damping,
                                  "relative_predicted_reduction":predicted, "rss_history":history[-10:],
                                  "jacobian":"analytic full-source formula derivatives", "jacobian_condition_number":condition},
                     extra={"formula":formula.source, "parameters":list(formula.parameters),
                            "columns":list(formula.columns), "start":dict(zip(formula.parameters, start.tolist(), strict=True)),
                            "constant_term":constant, "total_sum_of_squares":"centered" if constant else "uncentered"})
    result.provenance["optimizer"] = {"method":"Gauss-Newton with step halving and Levenberg-Marquardt damping",
                                       "tolerance":tolerance, "max_iterations":limit}
    result.spec = spec
    return result


@contextmanager
def _scratch():
    directory = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
    fd, name = tempfile.mkstemp(prefix="openecon-rreg-", suffix=".sqlite", dir=directory)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(name)
        connection.execute("PRAGMA cache_size=-4096")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE excluded (position INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE residuals (value REAL NOT NULL)")
        connection.execute("CREATE INDEX residual_order ON residuals(value)")
        yield connection, Path(name)
    except sqlite3.Error as error:
        raise AnalysisError("scratch_storage", "Robust regression needs writable scratch storage for exact global order statistics; check available disk space.") from error
    finally:
        if connection is not None:
            connection.close()
        for suffix in ["", "-journal", "-wal", "-shm"]:
            Path(name+suffix).unlink(missing_ok=True)


def _keep(db, positions):
    excluded = {row[0] for row in db.execute("SELECT position FROM excluded WHERE position BETWEEN ? AND ?", (int(positions[0]), int(positions[-1])))}
    return torch.tensor([position not in excluded for position in positions.tolist()], dtype=torch.bool)


def _median(db, expression="value", arguments=()):
    n = db.execute("SELECT COUNT(*) FROM residuals").fetchone()[0]
    if not n:
        raise AnalysisError("empty_sample", "No screened residuals remain.")
    # SQLite sorts on disk when the expression needs a temporary order. No
    # observation-length Python container is ever constructed.
    rows = db.execute(f"SELECT {expression} FROM residuals ORDER BY {expression} LIMIT ? OFFSET ?",
                      (*arguments, *arguments, 1 if n % 2 else 2, (n-1)//2)).fetchall()
    low, high = rows[0][0], rows[-1][0]
    total = low+high
    value = low if len(rows) == 1 else total/2 if math.isfinite(total) else low/2+high/2
    if not math.isfinite(value):
        raise AnalysisError("numerical_failure", "Residual order statistics exceed finite float64 precision.")
    return value


class _Screened:
    def __init__(self, sample, n, reporting, positions, positions_hash):
        self.base = sample
        self.nobs = self.nrows = n
        self.sample, self.sample_positions = reporting, positions
        self.positions_hash = positions_hash

    def __getattr__(self, name):
        return getattr(self.base, name)

    def provenance(self):
        record = self.base.provenance()
        record.update({"prescreen_positions_hash":record["sample_positions_hash"],
                       "sample_positions_hash":self.positions_hash,
                       "sample_hash":hashlib.sha256((record["data_hash"]+self.positions_hash).encode()).hexdigest(),
                       "sample_position_count":self.nrows, "sample_filter":"global initial OLS Cook distance <= 1 (undefined leverage-one distance retained)"})
        record["streaming"]["retained_reporting_rows"] = len(self.sample)
        return record


def _fit_rreg(spec, source, batch_rows):
    notes, sample = _Notes(spec), ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("x")
    sample.prepare()
    k = len(design.terms)
    if not k:
        raise AnalysisError("no_regressors", "No estimable robust-regression term remains.")
    if sample.nrows <= k:
        raise AnalysisError("insufficient_observations", "Robust regression needs more observations than parameters.")
    tune, tolerance = float(notes.option("tune")), float(notes.option("tolerance"))
    if not tune > 0 or not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tune must be positive; tolerance must lie between 0 and 1.")
    resource = sample.plan_rows("full-source robust regression", {"robust_factors":512*(k+1)**2,
                               "robust_SQLite_cache":4*1024**2}, 128*(k+6)).record()
    with _scratch() as (db, path):
        def original():
            for batch in sample.batches():
                yield batch.designs["x"], batch.numeric(spec.outcome), batch.weights
        factor, _ = _factor(original, k)
        beta, bread, _ = _solve(factor, k)
        initial_rss = float((factor[:, k]-factor[:, :k]@beta).square().sum())
        variance = initial_rss/(sample.nrows-k)
        if not math.isfinite(variance):
            raise AnalysisError("numerical_failure", "Initial OLS variance exceeds finite float64 precision; rescale the outcome.")
        if not variance > 0:
            raise AnalysisError("perfect_fit", "Initial least squares fits exactly; robust regression has nothing to downweight.")
        dropped, n, positions, chunks, digest = 0, 0, [], [], hashlib.sha256()
        for batch in sample.batches():
            x, y = batch.designs["x"], batch.numeric(spec.outcome)
            resid, leverage = y-x@beta, ((x@bread)*x).sum(1)
            _finite(resid, leverage)
            gap = 1-leverage
            distance = torch.where(gap > 1e-12, resid.square()*leverage/(k*variance*gap.square()), torch.zeros_like(gap))
            gross = distance > 1
            db.executemany("INSERT INTO excluded VALUES (?)", ((int(value),) for value in batch.positions[gross].tolist()))
            keep = ~gross
            dropped += int(gross.sum())
            n += int(keep.sum())
            kept_positions = batch.positions[keep]
            digest.update(kept_positions.numpy().astype("<i8", copy=False).tobytes())
            take = min(400-len(positions), len(kept_positions))
            if take:
                chunks.append(batch.frame.iloc[keep.nonzero().flatten().tolist()][:take].copy())
                positions.extend(kept_positions[:take].tolist())
        db.commit()
        if n <= k:
            raise AnalysisError("insufficient_observations", "Too few observations remain after the global Cook screen.")
        if dropped:
            notes.warn(f"Excluded {dropped} observation(s) with Cook's distance above 1 in the initial full-source least-squares fit.")
        selected = list(range(k))
        def blocks():
            for batch in sample.batches():
                keep = _keep(db, batch.positions)
                if bool(keep.any()):
                    yield batch, keep, batch.designs["x"][keep][:, selected], batch.numeric(spec.outcome)[keep]
        def screened_ols():
            for _, _, x, y in blocks():
                yield x, y, torch.ones_like(y)
        factor, _ = _factor(screened_ols, k)
        selected, omitted = kernel_call(collinear_columns, factor[:, :k])
        if omitted:
            sample.notes["omitted_terms"].extend(design.terms[index] for index in omitted)
            factor = torch.cat((factor[:, selected], factor[:, k:k+1]), 1)
        k = len(selected)
        beta, bread, condition = _solve(factor, k)
        counts, log, total_iterations = {"huber":0, "biweight":0}, [], 0
        c = robust.BIWEIGHT_C*tune/7
        prior = None
        maximum_y = 0.
        for _, _, _, y in blocks():
            maximum_y = max(maximum_y, float(y.abs().max()))
        for stage, threshold in [("huber", max(robust.HUBER_TOLERANCE, tolerance)), ("biweight", tolerance)]:
            while True:
                if total_iterations >= robust.MAX_ITERATIONS:
                    raise AnalysisError("nonconvergence", "Robust regression exhausted its full-source iteration limit.")
                total_iterations += 1
                counts[stage] += 1
                db.execute("DELETE FROM residuals")
                for _, _, x, y in blocks():
                    resid = y-x@beta
                    _finite(resid)
                    db.executemany("INSERT INTO residuals VALUES (?)", ((value,) for value in resid.tolist()))
                db.commit()
                median = _median(db)
                scale = _median(db, "abs(value-?)", (median,))/robust.MAD_CONSTANT
                if not scale > robust._ZERO_SCALE*maximum_y:
                    raise AnalysisError("zero_scale", "The global residual MAD is zero; robust weights are undefined.")
                used = beta.clone(), scale, stage
                change = 0.
                def reweighted():
                    nonlocal change
                    for _, _, x, y in blocks():
                        new = _case_weights(x, y, used, c)
                        old = torch.ones_like(y) if prior is None else _case_weights(x, y, prior, c)
                        change = max(change, float((new-old).abs().max()))
                        yield x, y, new
                factor, diagnostics = _factor(reweighted, k)
                beta, _, condition = _solve(factor, k)
                prior = used
                log.append({"stage":stage, "iteration":total_iterations, "max_weight_change":change})
                del log[:-robust._LOG_LIMIT]
                if change < threshold:
                    break
        sum_psi, sum_wr2, sum_w = _CompensatedSum(()), _CompensatedSum(()), _CompensatedSum(())
        low, high, zero, below, predictions = math.inf, -math.inf, 0, 0, []
        for batch, keep, x, y in blocks():
            fitted, weights = x@beta, _case_weights(x, y, prior, c)
            resid = y-fitted
            ratio = (resid/(scale*c)).square()
            psi = torch.where(ratio < 1, (1-ratio)*(1-5*ratio), torch.zeros_like(ratio))
            _finite(psi, weights*resid)
            sum_psi.add(psi.sum())
            sum_wr2.add((weights*resid).square().sum())
            sum_w.add(weights.sum())
            low, high = min(low, float(weights.min())), max(high, float(weights.max()))
            zero += int((weights == 0).sum())
            below += int((weights < .5).sum())
            take = min(400-len(predictions), len(y))
            for row, o, f in zip(batch.positions[keep][:take].tolist(), y[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row":row, "observed":o, "fitted":f, "residual":o-f})
        m = float(sum_psi.value)/n
        if not m > 0:
            raise AnalysisError("undefined_pseudovalues", "The global mean biweight derivative is not positive; increase tune.")
        correction = 1+k/(n-k)*(1-m)/m
        variance = (correction/m)**2*float(sum_wr2.value)/(n-k)
        if not variance > 0 or not math.isfinite(variance):
            raise AnalysisError("perfect_fit" if variance == 0 else "numerical_failure", "Pseudovalue variance is not positive finite.")
        transform = design.transform[selected][:, selected]
        raw_beta, covariance = transform@beta, transform@(bread*variance)@transform.T
        _finite(raw_beta, covariance)
        terms = [design.terms[index] for index in selected]
        slopes = [index for index, name in enumerate(terms) if name != "Intercept"]
        tests = {"model":wald_test(raw_beta, covariance, slopes, df_resid=n-k,
                                   label="F test of the slopes (pseudovalue regression)")} if slopes else {}
        reporting = pd.concat(chunks, ignore_index=True)
        proxy = _Screened(sample, n, reporting, positions, digest.hexdigest())
        diagnostics.update({"iterations":total_iterations, "converged":True,
                            "final_max_weight_change":log[-1]["max_weight_change"],
                            "design_condition_number":condition, "median_method":"exact global SQLite order statistics",
                            "scratch_bytes":path.stat().st_size, "scratch_removed_on_exit":True})
        return _result(proxy, terms=terms, beta=raw_beta, covariance=covariance,
                       info={"covariance":"nonrobust", "df_resid":n-k, "df_inference":n-k,
                             "correction":"Street, Carroll and Ruppert pseudovalue covariance"},
                       metrics={"rmse":math.sqrt(variance), "scale":scale, "iterations":total_iterations,
                                "huber_iterations":counts["huber"], "biweight_iterations":counts["biweight"],
                                "n_dropped_cooks":dropped, "df_model":len(slopes), "df_resid":n-k},
                       notes=notes, predictions=predictions, tests=tests, solver="huber_biweight_irls",
                       diagnostics=diagnostics, resource=resource,
                       extra={"tune":tune, "tolerance":tolerance, "huber_cutoff_in_mad":robust.HUBER_MAD_MULTIPLE,
                              "huber_c":robust.HUBER_MAD_MULTIPLE*robust.MAD_CONSTANT, "biweight_c":c,
                              "huber_tolerance":max(robust.HUBER_TOLERANCE, tolerance),
                              "weights":{"min":low, "mean":float(sum_w.value)/n, "max":high,
                                         "n_zero":zero, "n_below_half":below},
                              "pseudovalues":{"mean_psi_prime":m, "lambda":correction}, "iteration_log":log})


def _case_weights(x, y, state, c):
    beta, scale, stage = state
    u = (y-x@beta)/scale
    weights = robust.huber_weights(u) if stage == "huber" else robust.biweight_weights(u, c)
    _finite(weights)
    return weights


def fit_streaming_nonlinear(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None) -> ResultBundle:
    """Bounded full-source NL or rreg; unsupported estimators never collect."""
    if spec.estimator not in SUPPORTED:
        raise AnalysisError("streaming_unsupported", "This adapter supports only nonlinear LS and robust regression.")
    try:
        with torch.device("cpu"), torch.no_grad():
            return _fit_nl(spec, source, batch_rows) if spec.estimator == "nl" else _fit_rreg(spec, source, batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
