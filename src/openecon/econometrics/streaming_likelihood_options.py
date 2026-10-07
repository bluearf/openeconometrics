"""Global option contracts for replayed likelihoods, with bounded row buffers."""
from __future__ import annotations

from contextlib import closing
import math
from pathlib import Path
import sqlite3
import tempfile

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import OptimResult
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import encode_cluster_labels, numeric_values

from .replay_sample import ReplaySample


def resolve_sample_limits(spec, source, *, batch_rows=None):
    """Resolve min/max against the entire native estimation sample, before truncation.

    This extra pass is checked against preparation's projected raw-source hash.
    A bound from a first batch or a display sample is never used.
    """
    if spec.estimator not in {"tobit", "truncreg", "ivtobit"}:
        return spec, None
    options = dict(spec.options)
    lower, upper = options.get("ll_at_min", False), options.get("ul_at_max", False)
    if spec.estimator == "truncreg" and (lower or upper):
        raise AnalysisError("invalid_spec", "Native truncreg requires numeric truncation limits; sample-extreme flags are Tobit options.")
    if not lower and not upper and spec.estimator != "truncreg":
        return spec, None
    if lower and options.get("ll") is not None or upper and options.get("ul") is not None:
        raise AnalysisError("invalid_spec", "Give each normal limit as an explicit scalar or a sample extreme, not both.")
    probe = ReplaySample(spec, source, batch_rows=batch_rows)
    minimum, maximum = math.inf, -math.inf
    for frame, _, _ in probe._raw():
        values = numeric_values(frame[spec.outcome], spec.outcome)
        if len(values):
            minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
    if not math.isfinite(minimum):
        raise AnalysisError("empty_sample", "No positive-weight complete outcome remains for limit discovery.")
    if lower:
        options["ll"], options["ll_at_min"] = minimum, False
    if upper:
        options["ul"], options["ul_at_max"] = maximum, False
    if options.get("ll") is not None and options.get("ul") is not None and not options["ll"] < options["ul"]:
        raise AnalysisError("invalid_limits", "The globally discovered lower limit must be below the upper limit.")
    return spec.model_copy(update={"options": options}), probe.baseline


def validate_survival_subjects(sample):
    """Check arbitrary cross-batch subject intervals and count subjects on disk."""
    from .streaming_likelihood import _binary_y, _role

    spec = sample.spec
    id_column = (_role(spec, "id") or [None])[0]
    entry_column = (_role(spec, "entry") or [None])[0]
    failure_column = (_role(spec, "failure") or [None])[0]
    failures, exposure = _CompensatedSum(()), _CompensatedSum(())
    subjects = float(sample.nobs)
    scratch_bytes = 0
    sample.plan_rows("replay survival subject validation", {"SQLite_cache_and_cursor": 4*1024**2},
                     512*(len(sample.columns)+4))
    try:
        with tempfile.TemporaryDirectory(prefix="openecon-streg-subjects-") as scratch:
            path = Path(scratch)/"subjects.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                    db.execute("PRAGMA "+pragma)
                db.execute("CREATE TABLE records(subject BLOB,entry REAL,exit REAL,frequency REAL,position INTEGER)")
                for batch in sample.batches():
                    time = batch.numeric(spec.outcome)
                    entry = batch.numeric(entry_column) if entry_column else torch.zeros_like(time)
                    delta = _binary_y(batch, failure_column) if failure_column else torch.ones_like(time)
                    frequency = batch.weights if spec.weight_type == "fweight" else torch.ones_like(time)
                    failures.add(frequency@delta)
                    exposure.add(frequency@(time-entry))
                    if id_column:
                        keys = encode_cluster_labels(batch.frame[id_column])
                        db.executemany("INSERT INTO records VALUES(?,?,?,?,?)", zip(
                            keys, entry.tolist(), time.tolist(), frequency.tolist(), batch.positions.tolist(), strict=True))
                if id_column:
                    db.execute("CREATE INDEX subject_intervals ON records(subject,exit,position)")
                    overlap = db.execute("SELECT COUNT(*) FROM (SELECT entry,LAG(exit) OVER "
                                         "(PARTITION BY subject ORDER BY exit,position) AS previous FROM records) "
                                         "WHERE entry<previous").fetchone()[0]
                    if overlap:
                        raise AnalysisError("overlapping_records", f"{overlap} record(s) overlap an earlier record of the same subject.")
                    subjects = float(db.execute("SELECT SUM(replicas) FROM (SELECT MAX(frequency) AS replicas "
                                                "FROM records GROUP BY subject)").fetchone()[0])
                db.commit()
                scratch_bytes = path.stat().st_size
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError("survival_spill_failed", "Survival subject validation needs writable temporary storage and enough disk space.") from error
    return {"n_subjects": subjects, "n_failures": float(failures.value),
            "time_at_risk": float(exposure.value), "id_overlap_checked": bool(id_column),
            "subject_scratch_bytes": scratch_bytes}


def likelihood_covariance(sample, builder, theta, hessian, *, canonical_units=None):
    """Native GLM trial-prior centring and streg robust id-cluster covariance."""
    from .streaming_likelihood import _role
    from . import registry
    spec = sample.spec
    if (spec.estimator == "glm" and _role(spec, "trials")
            and spec.covariance == "cluster" and len(registry.cluster_columns(spec)) == 2):
        # Eigenvalue clipping of a two-way score meat is coordinate dependent.
        # Native GLM centres its design with the quasi-likelihood prior w*n,
        # whereas the generic ReplaySample centres with the user weights w.
        # Map the full score/Hessian to that original-unit, trial-centred basis
        # BEFORE PSD repair, then map its covariance back to replay coordinates.
        design = sample.designs["mean"]
        transform = torch.diag(1/design.scales[design.kept])
        if design.intercept:
            total, mass = _CompensatedSum((len(theta),)), _CompensatedSum(())
            for batch in sample.batches():
                prior = batch.weights*batch.numeric(_role(spec, "trials")[0])
                total.add(batch.designs["mean"].T@prior)
                mass.add(prior.sum())
            transform[0, 1:] = (total.value/mass.value)[1:]
        inverse = torch.linalg.solve_triangular(transform, torch.eye(len(theta), dtype=torch.float64), upper=True)
        class NativeScores:
            def __init__(self, native):
                self.native = native
            def score_rows(self, ignored):
                return self.native.score_rows(theta)@inverse
        covariance, inference = sample.covariance(lambda batch: NativeScores(builder(batch)), theta,
                                                  inverse.T@hessian@inverse)
        covariance = inverse@covariance@inverse.T
        inference["covariance_coordinate_basis"] = "original units centred with the binomial trial prior"
        return (covariance+covariance.T)/2, inference
    ids = _role(spec, "id") if spec.estimator == "streg" else []
    if spec.covariance != "robust" or not ids:
        return sample.covariance(builder, theta, hessian, canonical_units=canonical_units)
    sample.spec = spec.model_copy(update={"covariance": "cluster", "cluster": ids[0]})
    try:
        covariance, inference = sample.covariance(builder, theta, hessian, canonical_units=canonical_units)
    finally:
        sample.spec = spec
    inference.update({"covariance": "robust", "correction": "sandwich clustered on the id variable: G/(G-1)"})
    return covariance, inference


def glm_start(sample, builder):
    """Native weighted initial Fisher step and full-sample domain fallbacks."""
    width = len(sample.designs["mean"].terms)
    tree, prior_mass, initial_mean = _TSQRTree(), _CompensatedSum(()), _CompensatedSum(())
    weighted_valid, family, link = True, None, None
    for batch in sample.batches():
        obj = builder(batch).objective
        family, link = obj.family, obj.link
        mu = family.start_mu(obj.y)
        prior_mass.add(obj.prior.sum())
        initial_mean.add(obj.prior@mu)
        if not bool(family.valid_mu(mu).all()):
            weighted_valid = False
            continue
        eta = link.link(mu)
        if not bool(torch.isfinite(eta).all()) or not bool(link.valid(eta).all()):
            weighted_valid = False
            continue
        response = eta-(obj.offset if obj.offset is not None else 0.)
        weights = obj.prior*link.derivative(eta, mu).square()/family.variance(mu)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            weighted_valid = False
            continue
        tree.add(qr_factor(torch.cat((obj.x, response[:, None]), 1)*weights.sqrt()[:, None]))
    candidates = []
    if weighted_valid:
        factor = tree.finish()
        try:
            candidates.append(least_squares(factor[:, :width], factor[:, width], drop_collinear=False).beta)
        except Exception as error:
            if getattr(error, "code", None) != "singular_design":
                raise
    if sample.designs["mean"].intercept:
        average = initial_mean.value/prior_mass.value
        constant = torch.zeros(width, dtype=torch.float64)
        if bool(family.valid_mu(average)):
            constant[0] = link.link(average)
        candidates.append(constant)
    candidates.append(torch.zeros(width, dtype=torch.float64))
    for candidate in candidates:
        valid = bool(torch.isfinite(candidate).all())
        for batch in sample.batches():
            valid &= builder(batch).objective.state(candidate) is not None
        if valid:
            return candidate
    raise AnalysisError("invalid_start", "No full-sample IRLS starting candidate remains inside the native link domain.")


def fit_irls(sample, builder, start):
    """Global Fisher scoring by an augmented TSQR, matching native IRLS semantics.

    Working responses, Fisher weights and deviance are reduced over every row
    before a coefficient update. The covariance uses final expected information.
    """
    from .streaming_likelihood import _option

    width = len(start)
    tol, limit = _option(sample.spec, "tolerance", 1e-10), _option(sample.spec, "max_iterations", 100)
    scales = sample.designs["mean"].scales[sample.designs["mean"].kept]
    theta, steps, settled, previous = start.clone(), 0, False, math.inf
    converged, message = False, f"IRLS did not converge in {limit} global Fisher-scoring iterations."

    def value(candidate):
        deviance = _CompensatedSum(())
        for batch in sample.batches():
            objective = builder(batch).objective
            state = objective.state(candidate)
            if state is None:
                return None
            deviance.add(torch.tensor(state.deviance, dtype=torch.float64))
        return float(deviance.value)

    deviance = value(theta)
    if deviance is None:
        raise AnalysisError("invalid_start", "Global IRLS starting values are outside the link domain.")
    while True:
        tree = _TSQRTree()
        gradient, fisher = _CompensatedSum((width,)), _CompensatedSum((width, width))
        for batch in sample.batches():
            obj = builder(batch).objective
            state = obj.state(theta)
            if state is None:
                raise AnalysisError("numerical_failure", "The accepted global IRLS state left the link domain.")
            score, _, expected = obj.pieces(state)
            weights = obj.prior*expected
            response = state.eta-(obj.offset if obj.offset is not None else 0.)
            response = response+torch.where(expected > 0, score/expected.clamp_min(torch.finfo(torch.float64).tiny), 0.)
            if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(response).all()) or bool((weights < 0).any()):
                raise AnalysisError("numerical_failure", "The global IRLS working weights and response must be finite.")
            gradient.add(obj.x.T@(obj.prior*score))
            fisher.add(obj.x.T@(obj.x*weights[:, None]))
            tree.add(qr_factor(torch.cat((obj.x, response[:, None]), 1)*weights.sqrt()[:, None]))
        factor = tree.finish()
        try:
            fitted = least_squares(factor[:, :width], factor[:, width], drop_collinear=False)
        except Exception as error:
            if getattr(error, "code", None) != "singular_design":
                raise
            message = "The global IRLS working design became rank deficient as fitted weights vanished."
            break
        direction = fitted.beta-theta
        scaled = float(gradient.value@direction)
        # The native dense iteration centres the original-unit regressors.
        relative = float(((direction/scales).abs()/(theta/scales).abs().clamp_min(1.)).max())
        if settled and scaled <= tol and (relative <= tol or (relative <= math.sqrt(tol) and relative >= previous)):
            converged, message = True, "converged: global relative deviance change, scaled gradient and Fisher step"
            break
        if steps >= limit:
            break
        previous = relative
        allowance = 8*torch.finfo(torch.float64).eps*max(1., deviance)
        fraction, accepted = 1., None
        for _ in range(50):
            candidate = theta+fraction*direction
            candidate_deviance = value(candidate)
            if candidate_deviance is not None and candidate_deviance <= deviance+allowance:
                accepted = (candidate, candidate_deviance)
                break
            fraction *= .5
        if accepted is None:
            message = "Global IRLS step halving found no finite step with non-increasing deviance."
            break
        candidate, candidate_deviance = accepted
        settled = abs(candidate_deviance-deviance) <= tol*(abs(candidate_deviance)+1e-300)
        theta, deviance, steps = candidate, candidate_deviance, steps+1
    scale = sample.optimization_scale
    return OptimResult(theta, -.5*deviance*scale, gradient.value*scale, -fisher.value*scale,
                       steps, converged, "irls_fisher_scoring_tsqr",
                       {"message": message, "information": "expected", "global_updates": True})
