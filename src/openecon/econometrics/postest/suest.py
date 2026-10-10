"""Seemingly unrelated estimation (Stata's suest).

suest combines the estimates of several separately fitted models into one
parameter vector with a joint robust covariance, so that hypotheses across
models (equality of a coefficient in two equations) can be tested. With
per-observation scores ``s_mi`` and Hessians ``H_m`` of each model m,

    V = D (q sum_i s_i s_i') D,      D = blockdiag((-H_1)^-1, ..., (-H_M)^-1),

where ``s_i`` stacks the scores of observation i in every model (zero for a
model whose estimation sample does not contain i), the sum runs over the union
of the estimation samples (N observations) and ``q = N/(N-1)``. With
``cluster=`` the inner sum is over clusters of summed scores and
``q = G/(G-1)``. Each diagonal block is the Huber-White sandwich of that model
on its own sample (with the factor of the union).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
import torch

from openecon.analysis import _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import _versions, kernel_call
from openecon.econometrics.postest.common import (
    SUEST_FLAG, coefficients, is_suest, json_safe, matched_frame, require_result,
    result_covariance,
)
from openecon.econometrics.postest.scores import BINOMIAL_COMMANDS, SUPPORTED, model_scores
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines import covariance as cov
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes

MAX_BINOMIAL_JOINT_WORK = 500_000_000


def _binomial_joint_plan(bundles, matched):
    """Plan the complete dense joint system before reconstructing any scores."""
    if not any(bundle.spec.estimator in BINOMIAL_COMMANDS for bundle in bundles):
        return None
    counts = [len(rows) for _, rows in matched]
    widths = [len(bundle.coefficients) + (bundle.spec.estimator == "ols") for bundle in bundles]
    n, p = sum(counts), sum(widths)  # Conservative union upper bound; no row tensor yet.
    work = n * p * p + p ** 3
    if work > MAX_BINOMIAL_JOINT_WORK:
        raise AnalysisError("suest_work_limit", "Joint binomial suest reconstruction exceeds "
                            f"the {MAX_BINOMIAL_JOINT_WORK:,}-operation admission bound "
                            "(sum(N_m)*P^2 + P^3). Reduce model dimensions or model count.")
    return plan_workspace("joint binomial suest", {
        "retained_model_scores": sum(tensor_bytes((count, width)) for count, width in zip(counts, widths)) * 4,
        "joint_score_copies": tensor_bytes((n, p)) * 4,
        "joint_covariance_copies": tensor_bytes((p, p)) * 10,
        "likelihood_and_row_vectors": tensor_bytes((n,)) * 32,
    })


def _names(results: Sequence[ResultBundle], names: Sequence[str] | None) -> list[str]:
    if names is None:
        estimators = [r.spec.estimator for r in results]
        return [name if estimators.count(name) == 1 else f"{name}{i + 1}"
                for i, name in enumerate(estimators)]
    listed = None if isinstance(names, str) else list(names)
    if listed is None or len(listed) != len(results):
        raise AnalysisError("invalid_spec", "names must be a list with one name per result.")
    names = [str(name) for name in listed]
    if len(set(names)) != len(names) or any(not name or ":" in name for name in names):
        raise AnalysisError("invalid_spec", "names must be distinct, non-empty and free of ':'.")
    return names


@resident_cpu
def suest(*results: ResultBundle, data: Any, names: Sequence[str] | None = None,
          cluster: str | None = None) -> ResultBundle:
    """Seemingly unrelated estimation of several fitted models (Stata's ``suest``).

    Model
    -----
    Each model m is estimated separately by maximum likelihood (or least
    squares) on its own estimation sample; suest stacks the estimates into one
    vector and estimates their joint covariance by the sandwich formula

        V = D (q sum_i s_i s_i') D,    D = blockdiag((-H_1)^-1, ..., (-H_M)^-1),

    where ``s_i`` collects observation i's score contributions to every
    model (zero for a model that does not use the observation), ``H_m`` is
    model m's Hessian, the sum runs over the N observations in the union of
    the samples and ``q = N/(N-1)`` (Stata's ``_robust`` factor). With
    ``cluster='col'`` the scores are summed within clusters first and
    ``q = G/(G-1)``. The diagonal blocks are each model's robust covariance;
    the off-diagonal blocks are the cross-model covariances that make tests
    across models possible. Inference uses z statistics.

    Scores are rebuilt from the data (results do not store them): the design
    of each model is reconstructed from its specification, and the score and
    Hessian are evaluated at the reported estimates. Supported estimators
    include ``ols`` (as Stata's suest treats ``regress``: the
    Gaussian likelihood in ``(b, ln sigma^2)``, adding an equation
    ``<name>_lnvar`` with ``ln(SSR/N)``; the slope block equals HC0 times
    N/(N-1)), ``logit``, ``probit``, ``poisson`` (with offset/exposure),
    ``ologit``, ``oprobit``, ``mlogit``, ``glm``, ``nbreg``, ``tobit``,
    ``intreg``, ``truncreg``, ``cloglog`` and ``fracreg`` (logit/probit).
    The binomial commands require complete saved design/convergence metadata,
    and admit bounded resident CPU float64 score and joint buffers/work.
    All models must have the same weight type and
    values. fweights count virtual observations in the meat and N; pweights
    multiply scores; aweights normalize each model's likelihood weights.
    Anything else raises
    ``AnalysisError('suest_unsupported')``. The input fits may use any
    covariance: suest recomputes everything from the scores.

    Parameters
    ----------
    *results : ResultBundle
        Two or more fits, each on (a subset of) the rows of ``data``.
    data : DataFrame
        The dataset all models were fitted on (verified by hash).
    names : list of str, optional
        Model names used as equation names and term prefixes; default the
        estimator names (``ols``, ``logit``; numbered when repeated).
    cluster : str, optional
        Cluster column for the cluster-robust version (Stata ``vce(cluster)``).

    Returns
    -------
    A ResultBundle whose terms are ``<name>:<term>`` (``m1:x1``,
    ``m1:/lnvar``) grouped in equations ``<name>`` or ``<name>_<equation>``
    (``ols`` fits: ``<name>_mean`` and ``<name>_lnvar``), with the joint
    covariance in ``covariance_matrix``. ``nobs`` is the effective union size
    (sum of fweights, otherwise physical rows); ``spec`` is the first model's specification (the combined
    result cannot be refitted with ``oe.fit``); ``extra['models']`` lists the
    models; ``inference`` records ``covariance`` ('robust' or 'cluster'), the
    factor and the cluster count. Cross-model hypotheses are tested with this
    covariance, e.g. ``(b1 - b2)^2 / (V11 + V22 - 2 V12) ~ chi2(1)``.

    Errors: ``invalid_result``, ``invalid_spec`` (fewer than two results, bad
    names), ``data_mismatch`` (another dataset), ``suest_unsupported``,
    ``suest_mismatch`` (the score does not vanish at the estimates, e.g. a
    non-converged fit), ``missing_values``/``insufficient_clusters`` for the
    cluster column.

    Stata: ``estimates store m1`` ... ``suest m1 m2 [, vce(cluster id)]``.

    Example::

        m1 = oe.logit(data=df, y="union", x=["age", "grade"])
        m2 = oe.probit(data=df, y="union", x=["age", "grade"])
        joint = oe.suest(m1, m2, data=df, names=["lgt", "prb"])
        joint.covariance_matrix
    """
    if len(results) < 2:
        raise AnalysisError("invalid_spec", "suest needs at least two fitted results.")
    bundles = [require_result(value, f"result {i + 1}") for i, value in enumerate(results)]
    for bundle in bundles:
        if is_suest(bundle):
            raise AnalysisError("suest_unsupported", "A combined suest result cannot be combined "
                                "again; pass the individual fits.")
    labels = _names(bundles, names)
    matched = [matched_frame(bundle, data, f"model {label_name!r}")
               for label_name, bundle in zip(labels, bundles, strict=True)]
    joint_plan = _binomial_joint_plan(bundles, matched)
    frame = matched[0][0]
    pieces = [model_scores(bundle, frame, rows)
              for bundle, (_, rows) in zip(bundles, matched, strict=True)]
    union = torch.unique(torch.cat([piece.rows for piece in pieces]), sorted=True)
    n = len(union)
    weight_types = {bundle.spec.weight_type if bundle.spec.weights else None for bundle in bundles}
    if len(weight_types) != 1:
        raise AnalysisError("suest_unsupported", "All suest models must use the same weight type and values.")
    weight_type = next(iter(weight_types))
    frequency = None
    if weight_type is not None:
        common = frame[bundles[0].spec.weights].iloc[union.numpy()].to_numpy(dtype="float64")
        for bundle in bundles[1:]:
            other = frame[bundle.spec.weights].iloc[union.numpy()].to_numpy(dtype="float64")
            if not bool((common == other).all()):
                raise AnalysisError("suest_unsupported", "All suest models must use the same weight values on the union sample.")
        if weight_type == "fweight":
            frequency = torch.from_numpy(common.copy())
    effective_n = n if frequency is None else int(round(float(frequency.sum())))
    widths = [len(piece.terms) for piece in pieces]
    total = sum(widths)
    stacked = torch.zeros((n, total), dtype=torch.float64)
    column = 0
    for piece, width in zip(pieces, widths, strict=True):
        stacked[torch.searchsorted(union, piece.rows), column:column + width] = piece.scores
        column += width
    bread = torch.block_diag(*[piece.bread() for piece in pieces])
    inference: dict[str, Any] = {}
    if cluster is None:
        if effective_n < 2:
            raise AnalysisError("insufficient_observations", "suest needs at least two "
                                "observations.")
        factor = effective_n / (effective_n - 1)
        white = stacked if frequency is None else stacked / frequency.sqrt()[:, None]
        meat = kernel_call(cov.meat_white, white) * factor
        kind, correction = "robust", "suest sandwich: N/(N-1)"
    else:
        if not isinstance(cluster, str) or cluster not in frame.columns:
            raise AnalysisError("missing_columns", f"The cluster column {cluster!r} is not in the "
                                "data.")
        labels_column = frame[cluster].iloc[union.numpy()]
        if bool(labels_column.isna().any()):
            raise AnalysisError("missing_values", f"The cluster column {cluster!r} has missing "
                                "values in the estimation samples.")
        codes, levels = pd.factorize(labels_column, sort=False)
        groups = len(levels)
        if groups < 2:
            raise AnalysisError("insufficient_clusters", "Cluster covariance requires at least "
                                "two observed clusters.")
        factor = groups / (groups - 1)
        meat = kernel_call(cov.meat_cluster, stacked, torch.from_numpy(codes.astype("int64")),
                           groups) * factor
        kind, correction = "cluster", "suest cluster sandwich: G/(G-1)"
        inference.update({"cluster_count": groups, "cluster_column": cluster,
                          "cluster_df": groups - 1})
    covariance = kernel_call(cov.sandwich, bread, meat)
    params = torch.cat([piece.params for piece in pieces])
    terms, equations = [], []
    for label_name, piece in zip(labels, pieces, strict=True):
        for term, equation in zip(piece.terms, piece.equations, strict=True):
            terms.append(f"{label_name}:{term}")
            equations.append(label_name if equation is None else f"{label_name}_{equation}")
    first = bundles[0]
    alpha = first.spec.alpha
    rows = coefficients(first, params, covariance, alpha=alpha, df=None, terms=terms,
                        equations=equations)
    models = [{"name": label_name, "estimator": bundle.spec.estimator,
               "outcome": bundle.spec.outcome, "nobs": len(piece.rows),
               "n_parameters": len(piece.terms), "result_id": bundle.id,
               "covariance_of_input": result_covariance(bundle)}
              for label_name, bundle, piece in zip(labels, bundles, pieces, strict=True)]
    inference_record = {
        "covariance": kind, "use_t": False, "distribution": "normal", "alpha": alpha,
        "confidence_level": 1 - alpha, "df_resid": None, "df_inference": None,
        "cluster_count": None, "cluster_df": None, "cluster_column": None,
        "small_sample_correction": factor, "correction": correction,
        "intercept": first.spec.intercept, "n_parameters": len(rows),
        "residual_definition": "not applicable (combined estimation)",
        "weight_type": weight_type, "nobs_physical": n, "nobs_effective": effective_n,
        "weight_semantics": "frequency duplication" if frequency is not None else
                            "weighted likelihood scores; aweights normalized per model" if weight_type else "unweighted",
        **inference,
    }
    sample_hash = _frame_hasher(frame.iloc[union.numpy()].loc[:, list(dict.fromkeys(
        column for bundle in bundles for column in bundle.provenance.get("input_columns", [])))])
    provenance = {
        "schema_version": "3", "backend": "openecon.torch", "engine": "openecon", "device": "cpu",
        "estimator": "suest", "family": "postest", "stata_equivalent": ["suest"],
        "versions": _versions(), "data_hash": first.provenance.get("data_hash"),
        "sample_hash": sample_hash.hexdigest(), "precision": "float64",
        "sample_position_base": 0, "design_terms": terms, "stata_parity_validated": False,
        "postestimation": {"method": SUEST_FLAG, "models": models,
                           "score_construction": "design rebuilt from each spec; analytic "
                                                 "scores and Hessians at the reported estimates"},
    }
    if joint_plan is not None:
        provenance["resource_plan"] = joint_plan.record()
    notes = [f"Combined estimation of {len(bundles)} models on {n} observations (union of the "
             "estimation samples); the specification shown is the first model's."]
    if len({len(piece.rows) for piece in pieces}) > 1 or n != len(pieces[0].rows):
        notes.append("The models use different estimation samples; observations outside a "
                     "model's sample contribute zero scores to it.")
    return ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=first.spec,
        nobs=effective_n, nobs_original=len(frame), dropped_rows=len(frame) - n, coefficients=rows,
        covariance_matrix=((covariance + covariance.T) / 2).tolist(),
        metrics={"n_models": float(len(bundles))}, warnings=notes, predictions=[],
        sample_positions=union.tolist(), provenance=json_safe(provenance),
        inference=json_safe(inference_record), title="Seemingly unrelated estimation",
        tests={}, extra=json_safe({"models": models, "supported_estimators": list(SUPPORTED)}),
    )
