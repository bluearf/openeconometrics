"""Full-source PPML IRLS with disk-backed changing-weight FE projections."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import combinations
import math
import shutil
import sqlite3
import tempfile

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.count_numeric import poisson_deviance, poisson_logmass
from openecon.engines.covariance import nearest_psd
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .glm.common import check_pweights
from .glm.kernels import _DRIFT, _EPS, _MAX_HALVINGS, _STALL
from .replay_sample import ReplaySample
from .streaming_hdfe import _Selection, _Vectors, _absorb
from .streaming_linear import _Notes, _finite, _result, _scratch_directory

SUPPORTED = frozenset({"ppmlhdfe"})


def supports(spec):
    return spec.estimator in SUPPORTED


def _offset(sample, batch):
    offsets, exposures = (
        registry.role_columns(sample.spec, "offset"),
        registry.role_columns(sample.spec, "exposure"),
    )
    if offsets and exposures:
        raise AnalysisError("invalid_spec", "Give offset or exposure, not both.")
    if offsets:
        return batch.numeric(offsets[0])
    if exposures:
        v = batch.numeric(exposures[0])
        if bool((v <= 0).any()):
            raise AnalysisError("invalid_exposure", "Poisson exposure must be strictly positive.")
        return v.log()
    return torch.zeros(len(batch.frame), dtype=torch.float64)


def _prune(first, selection, dimensions, clusters, notes):
    selection.seed(first, dimensions, clusters, False)
    db = selection.connection
    db.execute("CREATE TABLE positive(position INTEGER PRIMARY KEY,value INTEGER)")
    for batch in first.batches():
        y = batch.numeric(first.spec.outcome)
        if bool((y < 0).any()):
            raise AnalysisError(
                "invalid_count_outcome", "Poisson pseudo-likelihood requires nonnegative outcomes."
            )
        _offset(first, batch)
        db.executemany(
            "INSERT INTO positive VALUES(?,?)",
            zip(batch.positions.tolist(), (y > 0).to(torch.int64).tolist(), strict=True),
        )
    separated, singletons = 0, 0
    while True:
        changed = 0
        for d in range(len(dimensions)):
            db.execute("DROP TABLE IF EXISTS zero_levels")
            db.execute(
                f"CREATE TEMP TABLE zero_levels AS SELECT f{d} AS key FROM rows JOIN positive USING(position) WHERE rows.alive=1 GROUP BY f{d} HAVING MAX(positive.value)=0"
            )
            db.execute("CREATE INDEX zero_level_key ON zero_levels(key)")
            db.execute(
                f"UPDATE rows SET alive=0 WHERE alive=1 AND f{d} IN(SELECT key FROM zero_levels)"
            )
            n = db.execute("SELECT changes()").fetchone()[0]
            separated += n
            changed += n
        if not db.execute("SELECT COUNT(*) FROM rows WHERE alive=1").fetchone()[0]:
            raise AnalysisError(
                "constant_outcome", "All outcomes are zero in the retained fixed-effect sample."
            )
        if notes.option("drop_singletons"):
            for d in range(len(dimensions)):
                db.execute("DROP TABLE IF EXISTS singleton")
                db.execute(
                    f"CREATE TEMP TABLE singleton AS SELECT f{d} AS key FROM rows WHERE alive=1 GROUP BY f{d} HAVING COUNT(*)+SUM(protected)=1"
                )
                db.execute("CREATE INDEX singleton_key ON singleton(key)")
                db.execute(
                    f"UPDATE rows SET alive=0 WHERE alive=1 AND f{d} IN(SELECT key FROM singleton)"
                )
                n = db.execute("SELECT changes()").fetchone()[0]
                singletons += n
                changed += n
            if not db.execute("SELECT COUNT(*) FROM rows WHERE alive=1").fetchone()[0]:
                raise AnalysisError(
                    "empty_sample",
                    "Every retained observation is a singleton in an absorbed level.",
                )
        if not changed:
            break
    selection.levels = []
    for d in range(len(dimensions)):
        db.execute(f"DELETE FROM keep{d}")
        db.execute(f"INSERT INTO keep{d} SELECT DISTINCT f{d} FROM rows WHERE alive=1")
        selection.levels.append(db.execute(f"SELECT COUNT(*) FROM keep{d}").fetchone()[0])
    selection.used = db.execute("SELECT COUNT(*) FROM rows WHERE alive=1").fetchone()[0]
    db.commit()
    if separated:
        notes.warn(
            f"Dropped {separated} observation(s) in fixed-effect levels whose outcomes are all zero (separated by a fixed effect)."
        )
    if singletons:
        notes.warn(
            f"Dropped {singletons} singleton observation(s) alone in a level of an absorbed dimension (drop_singletons=False keeps them)."
        )
    return separated, singletons


class _States(_Vectors):
    def __init__(self, sample, width, dimensions, metadata):
        parent = _scratch_directory()
        extra = sample.nrows * width * 8 * 20 + metadata * 2
        if extra + 64 * 1024**2 > shutil.disk_usage(parent or tempfile.gettempdir()).free:
            raise AnalysisError(
                "fixed_effect_disk_limit",
                "PPML IRLS and fixed-effect vectors need more free owned temporary disk space.",
            )
        super().__init__(sample, width, dimensions, metadata)
        self.maximum_files, self.required_bytes = 20, extra

    def scalar(self, stream, values):
        block = torch.zeros((len(values), self.width), dtype=torch.float64)
        block[:, 0] = values
        self.write(stream, block)


class _Working:
    def __init__(self, sample, states):
        self.sample, self.states, self.weight_mean = sample, states, 1.0

    def __getattr__(self, name):
        return getattr(self.sample, name)

    def batches(self):
        for batch, (state,) in self.states.batches(self.sample, "eta"):
            mu = state[:, 0].exp()
            batch.weights = batch.weights * mu
            if not bool(torch.isfinite(batch.weights).all()) or bool((batch.weights <= 0).any()):
                raise AnalysisError(
                    "precision_unsupported",
                    "IRLS working weights are not positive finite float64; rescale the outcome or weight units.",
                )
            yield batch


def _step(sample, states, dimensions, selected, tolerance):
    working = _Working(sample, states)
    with states.writer("raw") as writer:
        for batch, (eta,) in states.batches(sample, "eta"):
            value = eta[:, 0]
            mu = value.exp()
            z = value - _offset(sample, batch) + (batch.numeric(sample.spec.outcome) - mu) / mu
            states.write(writer, torch.cat((z[:, None], batch.designs["x"]), 1))
    sweeps, _, _ = _absorb(states, working, dimensions, min(1e-10, tolerance / 10), 10000)
    tree = _TSQRTree()
    for batch, (values,) in states.batches(working, "y"):
        x, z = values[:, [i + 1 for i in selected]], values[:, 0]
        tree.add(
            torch.linalg.qr(
                torch.cat((x, z[:, None]), 1) * batch.weights.sqrt()[:, None], mode="r"
            )[1]
        )
    factor = tree.finish()
    k = len(selected)
    fit = least_squares(factor[:, :k], factor[:, k], drop_collinear=False, tol=0.0)
    with states.writer("proposed_eta") as writer:
        for batch, (raw, within) in states.batches(sample, "raw", "y"):
            eta = (
                _offset(sample, batch)
                + raw[:, 0]
                - within[:, 0]
                + within[:, [i + 1 for i in selected]] @ fit.beta
            )
            states.scalar(writer, eta)
    return fit, sweeps


def _trial(sample, states, fraction):
    total = _CompensatedSum(())
    maximum, drifting, separated = 0.0, 0, 0
    finite = True
    with states.writer("candidate_eta") as writer:
        for batch, (old, new) in states.batches(sample, "eta", "proposed_eta"):
            change = fraction * (new[:, 0] - old[:, 0])
            eta = old[:, 0] + change
            mu = eta.exp()
            y = batch.numeric(sample.spec.outcome)
            finite &= bool(torch.isfinite(mu).all()) and bool((mu > 0).all())
            if finite:
                total.add((batch.weights * poisson_deviance(y, mu, eta=eta)).sum())
            states.scalar(writer, eta)
            maximum = max(maximum, float(change.abs().max()))
            moving = change.abs() > _DRIFT
            drifting += int(moving.sum())
            separated += int((moving & (y == 0)).sum())
    return float(total.value) if finite else math.inf, maximum, drifting, separated


def _covariance(sample, states, selected, bread, k_total, scales):
    kind, k = sample.spec.covariance, len(selected)
    info = {
        "covariance": kind,
        "df_inference": None,
        "df_resid": sample.nobs - k_total,
        "small_sample_correction": None,
    }
    if kind == "nonrobust":
        info["correction"] = (
            "observed information of the Poisson likelihood on the partialled-out regressors"
        )
        return bread, info
    columns = registry.cluster_columns(sample.spec) if kind == "cluster" else []
    subsets = [
        combo
        for size in range(1, len(columns) + 1)
        for combo in combinations(range(len(columns)), size)
    ]
    meat = _CompensatedSum(bread.shape)
    with ExitStack() as stack:
        accs = []
        for _ in subsets:
            acc = ClusterAccumulator(k, scratch_directory=_scratch_directory())
            stack.callback(acc.close)
            accs.append(acc)
        for batch, (values, eta) in states.batches(sample, "y", "eta"):
            x = values[:, [i + 1 for i in selected]]
            resid = batch.numeric(sample.spec.outcome) - eta[:, 0].exp()
            if kind == "robust":
                weights = (
                    batch.weights.sqrt() if sample.spec.weight_type == "fweight" else batch.weights
                )
                score = x * (weights * resid)[:, None]
                _finite(score)
                meat.add(score.T @ score)
            elif kind == "cluster":
                scores = x * (batch.weights * resid)[:, None]
                _finite(scores)
                labels = [encode_cluster_labels(batch.frame[name]) for name in columns]
                for combo, acc in zip(subsets, accs, strict=True):
                    keys = (
                        labels[combo[0]]
                        if len(combo) == 1
                        else [
                            b"".join(len(key).to_bytes(8, "big") + key for key in cell)
                            for cell in zip(*(labels[i] for i in combo), strict=True)
                        ]
                    )
                    acc.add(keys, scores)
            else:
                raise AnalysisError(
                    "unsupported_covariance",
                    "PPML supports nonrobust, robust or declared one/two-way cluster covariance.",
                )
        n = sample.nobs
        if kind == "robust":
            factor = n / (n - k_total)
            matrix = meat.value * factor
            info.update(
                {
                    "correction": "HC1: N/(N-K), K includes absorbed degrees of freedom",
                    "small_sample_correction": factor,
                }
            )
        else:
            counts = []
            for combo, acc in zip(subsets, accs, strict=True):
                matrix, g = acc.finish()
                meat.add(matrix * (1 if len(combo) % 2 else -1))
                if len(combo) == 1:
                    counts.append(g)
            matrix = meat.value
            adjusted = False
            if len(columns) > 1:
                matrix, adjusted = nearest_psd(matrix * scales[:, None] * scales[None, :])
                matrix = matrix / scales[:, None] / scales[None, :]
            g = min(counts)
            factor = g / (g - 1) * (n - 1) / (n - k_total)
            matrix *= factor
            info.update(
                {
                    "correction": "cluster sandwich: G/(G-1)*(N-1)/(N-K)"
                    if len(columns) == 1
                    else "multiway cluster sandwich (inclusion-exclusion): G_min/(G_min-1)*(N-1)/(N-K)",
                    "cluster_count": g,
                    "cluster_counts": counts,
                    "cluster_columns": columns,
                    "cluster_column": columns[0],
                    "cluster_df": g - 1,
                    "small_sample_correction": factor,
                    "psd_adjusted": adjusted,
                }
            )
        covariance = bread @ matrix @ bread
        return (covariance + covariance.T) / 2, info


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported", "No native PPML FE replay adapter is registered."
        )
    try:
        with torch.no_grad(), torch.device("cpu"), ExitStack() as stack:
            notes = _Notes(spec)
            check_pweights(notes)
            tolerance, maxiter = (
                float(notes.option("tolerance")),
                int(notes.option("max_iterations")),
            )
            if not 0 < tolerance < 1:
                raise AnalysisError(
                    "invalid_option", "PPML tolerance must lie strictly between zero and one."
                )
            dimensions = registry.role_columns(spec, "absorb")
            if not dimensions:
                raise AnalysisError(
                    "invalid_spec", "PPMLHDFE needs at least one absorbed dimension."
                )
            clusters = registry.cluster_columns(spec) if spec.covariance == "cluster" else []
            first = ReplaySample(spec, source, batch_rows=batch_rows).prepare()
            first.plan_rows(
                "PPML FE zero/singleton discovery",
                {"selection_SQLite_cache": 2 * 1024**2},
                256 * (len(dimensions) + 4),
            )
            selection = _Selection(len(dimensions), len(clusters))
            stack.callback(selection.close)
            separated, singletons = _prune(first, selection, dimensions, clusters, notes)
            dropped = first.nrows - selection.used
            sample = ReplaySample(
                spec,
                source,
                batch_rows=batch_rows,
                row_filter=(lambda frame: selection.filter(frame, dimensions)) if dropped else None,
            )
            design = sample.add_design("x", intercept=False, implicit_constant=True)
            sample.prepare()
            if (
                sample.baseline["data_hash"] != first.baseline["data_hash"]
                or sample.baseline["positions_hash"] != selection.position_hash()
            ):
                raise AnalysisError(
                    "source_changed",
                    "PPML source or zero/singleton-selected row identities changed during replay.",
                )
            width = len(design.terms)
            if not width:
                raise AnalysisError("empty_design", "No regressor remains after global screening.")
            resource = sample.plan_rows(
                "PPML changing-weight global projection and IRLS",
                {
                    "initial_reporting_sample": first.reporting_bytes,
                    "initial_category_metadata": first._category_bytes,
                    "selection_SQLite_cache": 2 * 1024**2,
                    "projection_SQLite_cache": 2 * 1024**2,
                    "global_IRLS_factors": 1024 * (width + 1) ** 2,
                    "PPML_cluster_caches": 18 * 1024**2 if clusters else 0,
                },
                384 * (width + 4) + 128 * len(dimensions),
            )
            states = _States(sample, width + 1, len(dimensions), selection.path.stat().st_size)
            stack.callback(states.close)
            before = _CompensatedSum((width,))
            with states.writer("raw") as writer:
                for batch in sample.batches():
                    before.add((batch.designs["x"].square() * batch.weights[:, None]).sum(0))
                    states.write(
                        writer,
                        torch.cat(
                            (
                                torch.zeros((len(batch.frame), 1), dtype=torch.float64),
                                batch.designs["x"],
                            ),
                            1,
                        ),
                    )
            _absorb(states, sample, dimensions, min(1e-10, tolerance / 10), 10000)
            tree = _TSQRTree()
            after = _CompensatedSum((width,))
            for batch, (values,) in states.batches(sample, "y"):
                after.add((values[:, 1:].square() * batch.weights[:, None]).sum(0))
                tree.add(
                    torch.linalg.qr(values[:, 1:] * batch.weights.sqrt()[:, None], mode="r")[1]
                )
            factor = tree.finish()
            absorbed = (after.value <= 1e-13 * before.value).nonzero().flatten().tolist()
            remaining = [i for i in range(width) if i not in absorbed]
            kept, omitted = collinear_columns(factor[:, remaining]) if remaining else ([], [])
            selected = [remaining[i] for i in kept]
            for index in [*absorbed, *(remaining[i] for i in omitted)]:
                notes.warn(
                    f"Omitted {design.terms[index]}: collinearity with absorbed fixed effects."
                )
                sample.notes["omitted_terms"].append(design.terms[index])
            if not selected:
                raise AnalysisError(
                    "empty_design", "Every regressor is absorbed by the declared fixed effects."
                )
            levels, redundant, nested, total = selection.degrees_of_freedom()
            constant_added = all(nested)
            df_absorbed = total + int(constant_added)
            k_total = len(selected) + df_absorbed
            if sample.nobs <= k_total:
                raise AnalysisError(
                    "insufficient_observations",
                    "PPML needs residual degrees of freedom after slopes and absorbed effects.",
                )
            moments = _WeightedMoments(2, intercept=True)
            for batch in sample.batches():
                y = batch.numeric(spec.outcome)
                moments.add(torch.stack((torch.ones_like(y), y), 1), batch.weights, False)
            mean = float(moments.anchor[1] + moments.magnitude[1] * moments.mean[1])
            if not mean > 0:
                raise AnalysisError(
                    "constant_outcome", "The positive-weight mean Poisson outcome is zero."
                )
            deviance = _CompensatedSum(())
            with states.writer("eta") as writer:
                for batch in sample.batches():
                    y = batch.numeric(spec.outcome)
                    mu = (y + mean) / 2
                    states.scalar(writer, mu.log())
                    deviance.add((batch.weights * poisson_deviance(y, mu)).sum())
            deviance = float(deviance.value)
            beta = torch.zeros(len(selected), dtype=torch.float64)
            stalled, sweeps = 0, 0
            converged = False
            for iteration in range(1, maxiter + 1):
                fit, used = _step(sample, states, dimensions, selected, tolerance)
                sweeps += used
                limit = math.inf if iteration == 1 else deviance + 8 * _EPS * max(1.0, deviance)
                fraction = 1.0
                for _ in range(_MAX_HALVINGS):
                    candidate, maximum, drifting, moving_zero = _trial(sample, states, fraction)
                    if math.isfinite(candidate) and candidate <= limit:
                        break
                    fraction *= 0.5
                else:
                    raise AnalysisError(
                        "nonconvergence",
                        "PPML step halving found no finite improving global likelihood step.",
                    )
                beta += fraction * (fit.beta - beta)
                change = abs(candidate - deviance)
                deviance = candidate
                states.swap("candidate_eta", "eta")
                if iteration > 1 and change <= tolerance * (abs(deviance) + 1e-300):
                    if maximum <= _DRIFT:
                        converged = True
                        break
                    stalled += 1
                    if stalled >= _STALL:
                        if moving_zero:
                            raise AnalysisError(
                                "separation_detected",
                                "The complete PPML deviance is flat while zero-outcome predictors keep drifting; separation by regressors/FE combinations is refused.",
                            )
                        raise AnalysisError(
                            "nonconvergence",
                            "PPML predictors keep drifting despite a flat global deviance.",
                        )
                else:
                    stalled = 0
            if not converged:
                raise AnalysisError(
                    "nonconvergence",
                    "PPML did not converge within its declared IRLS iteration limit.",
                )
            final, used = _step(sample, states, dimensions, selected, tolerance)
            sweeps += used
            scales = design.scales[design.kept][selected]
            covariance, info = _covariance(sample, states, selected, final.xtx_inv, k_total, scales)
            transform = torch.diag(1 / scales)
            raw_beta, raw_covariance = transform @ beta, transform @ covariance @ transform.T
            _finite(raw_beta, raw_covariance)
            total = _CompensatedSum((2,))
            for batch in sample.batches():
                exposure = _offset(sample, batch).exp()
                total.add(
                    torch.stack(
                        (
                            (batch.weights * batch.numeric(spec.outcome)).sum(),
                            (batch.weights * exposure).sum(),
                        )
                    )
                )
            constant = total.value[0] / total.value[1]
            ll, null = _CompensatedSum(()), _CompensatedSum(())
            predictions = []
            for batch, (eta,) in states.batches(sample, "eta"):
                y, mu = batch.numeric(spec.outcome), eta[:, 0].exp()
                nullmu = _offset(sample, batch).exp() * constant
                ll.add((batch.weights * poisson_logmass(y, mu)).sum())
                null.add((batch.weights * poisson_logmass(y, nullmu)).sum())
                take = min(400 - len(predictions), len(y))
                predictions.extend(
                    {"observed": float(a), "predicted": float(b), "residual": float(a - b)}
                    for a, b in zip(y[:take], mu[:take], strict=True)
                )
            ll, null = float(ll.value), float(null.value)
            if not all(math.isfinite(v) for v in (ll, null, deviance)):
                raise AnalysisError(
                    "precision_unsupported",
                    "Global Poisson likelihood or deviance exceeds finite float64.",
                )
            info.update(
                {
                    "nobs": sample.nobs,
                    "k_total": k_total,
                    "absorbed_degrees_of_freedom": df_absorbed,
                    "degrees_of_freedom_convention": "reghdfe: observed levels minus connected-component redundancy; dimensions nested within any declared cluster are omitted, with one constant added when all are nested",
                }
            )
            extra = {
                "absorbed": [
                    {"column": name, "levels": n, "redundant": r, "nested": inside}
                    for name, n, r, inside in zip(
                        dimensions, levels, redundant, nested, strict=True
                    )
                ],
                "constant_degree_of_freedom_added": constant_added,
                "null_log_likelihood": null,
                "null_model": "constant only (with the offset)",
                "log_likelihood_note": "Poisson log pseudolikelihood",
                "separation_check": "fixed-effect levels with all-zero outcomes only; general separation refused during IRLS",
                "demeaning_sweeps": sweeps,
                "tolerance": tolerance,
            }
            result = _result(
                sample,
                terms=[design.terms[i] for i in selected],
                beta=raw_beta,
                covariance=raw_covariance,
                info=info,
                metrics={
                    "pseudo_r_squared": 1 - ll / null if null != 0 else None,
                    "log_likelihood": ll,
                    "deviance": deviance,
                    "df_resid": sample.nobs - k_total,
                    "df_absorbed": df_absorbed,
                    "iterations": iteration,
                    "n_separated_dropped": separated,
                    "n_singletons_dropped": singletons,
                },
                notes=notes,
                predictions=predictions,
                tests={
                    "model": wald_test(
                        raw_beta,
                        raw_covariance,
                        list(range(len(selected))),
                        label="Wald chi2 test of the slopes",
                    )
                },
                solver="native_replayed_IRLS_and_weighted_disk_FE_TSQR",
                diagnostics={
                    "converged": True,
                    "iterations": iteration,
                    "demeaning_sweeps": sweeps,
                    **selection.diagnostics(),
                    **states.diagnostics(),
                },
                extra=extra,
                resource=resource.record(),
                use_t=False,
            )
            def fitted_blocks():
                for batch,(eta,) in states.batches(sample,'eta'):
                    yield batch.frame,eta[:,0].exp()
            from .postest.group_state import capture_fixed_replay
            result.extra['group_state']=capture_fixed_replay(result,fitted_blocks)
            return result
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "ppml_spill_failed",
            "PPML replay needs writable owned temporary storage and enough free disk space.",
        ) from error
