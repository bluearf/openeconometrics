"""Explicit profile and whole-respondent bootstrap laws with complete replay."""

from __future__ import annotations

import math
import random

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from . import common as c
from .model import confidence
from .post import conjoint_importance, conjoint_simulate


def options(replications, seed, level):
    return (c.c.check_count(replications, "replications", minimum=20, maximum=1999),
            c.c.check_count(seed, "seed", minimum=0, maximum=2**32-1), confidence(level))


def moments(values, point, level):
    """Full empirical covariance and marginal linear-interpolated percentiles."""
    mean = values.mean(0)
    centered = values-mean
    covariance = centered.T@centered/(len(values)-1)
    limits = torch.quantile(values, torch.tensor([(1-level)/2, (1+level)/2], dtype=torch.float64),
                          dim=0, interpolation="linear")
    if not all(bool(torch.isfinite(a).all()) for a in (values, point, covariance, limits)):
        raise AnalysisError("numerical_failure", "Bootstrap targets/covariance/intervals are nonfinite.")
    rows = [[float(point[i]), math.sqrt(float(covariance[i, i])), float(mean[i]),
             float(mean[i]-point[i]), float(limits[0, i]), float(limits[1, i]),
             bool(covariance[i, i] == 0)] for i in range(len(point))]
    return rows, covariance


MOMENT_COLUMNS = ["estimate", "std_error", "bootstrap_mean", "bias", "ci_low", "ci_high", "degenerate"]


def sealed(procedure, tables, state, settings):
    output = c.seal(procedure, tables, state,
                    inference="approximate marginal percentile bootstrap; no joint band, t, p or exact coverage",
                    **settings)
    c.conjoint_save(output)  # The complete, untruncated result must fit the artifact bound.
    return output


def qr(design):
    singular = torch.linalg.svdvals(design)
    threshold = max(design.shape)*torch.finfo(torch.float64).eps*float(singular[0])
    if float(singular[-1]) <= threshold or float(singular[0]/singular[-1]) > 1e10:
        raise AnalysisError("unidentified_replica", "A bootstrap design is rank deficient/ill-conditioned; no redraw or repair.")
    return torch.linalg.qr(design, mode="reduced")


def refit(q, r, response):
    origin = response.mean()
    beta = torch.linalg.solve_triangular(r, (q.T@(response-origin))[:, None], upper=True).flatten()
    beta[0] += origin
    return beta


@resident_cpu
def conjoint_bootstrap_fit(result, *, method="residual", replications=499, seed=1729, level=0.95,
                           max_work=100_000_000, device="cpu", weights=None):
    """Refit scored conjoint under residual, HC2 wild or iid random-profile pairs laws.

    Residual resampling assumes independent homoskedastic errors at fixed profiles.
    Rademacher/Mammen wild laws use e/sqrt(1-h), independent profile errors and fixed X.
    Pairs resampling assumes iid random (X,y) rows, not a fixed experimental plan.
    Every draw and full coefficient/utility covariance is saved. Any unidentified
    replica fails the entire operation. CR fits need a separate block law and fail.
    Marginal percentile intervals are approximate, without bootstrap-t or p-values.
    """
    state = c.intact(result, "conjoint_fit")
    c.c.check_choice(method, "method", ("residual", "rademacher", "mammen", "pairs"))
    b, seed, level = options(replications, seed, level)
    if state["covariance"] in ("cr0", "cr1"):
        raise AnalysisError("unsupported_dependence", "Independent-profile bootstrap cannot reuse a clustered fit; declare a separate block law.")
    n, p, m = len(state["training_profile_ids"]), len(state["terms"]), len(state["subjects"])
    k = sum(len(v) for v in state["attributes"].values())
    d = p+k
    work = m*b*(3*n*p*p+3*p**3+d*p+d*d) + m*d*b*max(1, math.ceil(math.log2(b)))
    settings = c.guard("conjoint_bootstrap_fit", n, p, m, device=device, weights=weights,
                       max_work=max_work, work=work, extra_buffers={
                           "all_draws_replicas_and_serialized_tables": 256*m*b*(n+d),
                           "full_joint_target_covariance_and_output": 256*m*d*d,
                           "scaled_target_map": 8*d*p,
                       })
    attributes = {a: state["attributes"][a] for a in state["attribute_order"]}
    design, terms, _, _, _ = c.coding(result["training_plan"], attributes, state["modes"],
                                     anchors=state["anchors"])
    if terms != state["terms"]:
        raise AnalysisError("invalid_saved_result", "Saved coefficient order differs from the training design.")
    q, r = qr(design)
    complement = 1-torch.sum(q*q, dim=1)
    if method in ("rademacher", "mammen") and bool((complement <= 1e-12).any()):
        raise AnalysisError("unit_leverage", "HC2 wild resampling requires every profile leverage below one.")
    transform = torch.tensor(state["transform"], dtype=torch.float64)
    metadata = [[f"target_{i}", "coefficient", term, None, None] for i, term in enumerate(terms)]
    target_map = [row for row in transform]
    for item in state["mapping"]:
        for value in state["attributes"][item["attribute"]]:
            metadata.append([f"target_{len(metadata)}", "utility", None, item["attribute"], value])
            target_map.append(c.level_vector(item, value, state["attributes"][item["attribute"]], p)@transform)
    mapping = torch.stack(target_map)
    names = [row[0] for row in metadata]
    fitted_rows = {(c.key(row[0]), c.key(row[1])): row[2:]
                   for row in result["fitted"].itertuples(index=False, name=None)}
    rng = random.Random(seed)
    estimates, covariances, replicas, provenance = [], [], [], []
    for record in state["subjects"]:
        who = record["subject"]
        sample = torch.tensor([fitted_rows[(c.key(who), c.key(card))]
                               for card in state["training_profile_ids"]], dtype=torch.float64)
        observed, fitted, residual = sample.T
        beta = torch.tensor(record["parameters_scaled"], dtype=torch.float64)
        if method == "residual":
            errors = (residual-residual.mean())*math.sqrt(n/(n-p))
        elif method != "pairs":
            errors = residual/torch.sqrt(complement)
        draws, targets = [], []
        for replicate in range(b):
            if method in ("pairs", "residual"):
                draw = [rng.randrange(n) for _ in range(n)]
                if method == "pairs":
                    index = torch.tensor(draw, dtype=torch.int64)
                    try:
                        qq, rr = qr(design[index])
                    except AnalysisError as error:
                        raise AnalysisError(error.code, f"Subject {who!r}, replica {replicate}: {error}") from error
                    coefficient = refit(qq, rr, observed[index])
                else:
                    coefficient = refit(q, r, fitted+errors[draw])
            else:
                if method == "rademacher":
                    draw = [1.0 if rng.random() < .5 else -1.0 for _ in range(n)]
                else:
                    root = math.sqrt(5)
                    draw = [(1-root)/2 if rng.random() < (root+1)/(2*root) else (1+root)/2
                            for _ in range(n)]
                coefficient = refit(q, r, fitted+errors*torch.tensor(draw, dtype=torch.float64))
            targets.append(mapping@coefficient)
            draws.append(draw)
        targets = torch.stack(targets)
        rows, covariance = moments(targets, mapping@beta, level)
        estimates.extend([who, *meta, *values] for meta, values in zip(metadata, rows))
        covariances.extend([who, name, *covariance[i].tolist()] for i, name in enumerate(names))
        replicas.extend([who, i, *values] for i, values in enumerate(targets.tolist()))
        provenance.append({"subject": who, "source_positions": record["sample_positions"],
                           "source_labels": record["sample_labels"], "observed": observed.tolist(),
                           "fitted": fitted.tolist(), "residual": residual.tolist(),
                           "parameters_scaled": beta.tolist(), "draws": draws})
    assumption = {"residual": "fixed profiles, independent homoskedastic errors; centered residuals scaled sqrt(n/(n-p))",
                  "rademacher": "fixed profiles, independent heteroskedastic errors; HC2 residuals; symmetric unit multipliers",
                  "mammen": "fixed profiles, independent heteroskedastic errors; HC2 residuals; Mammen unit-variance/third-moment multipliers",
                  "pairs": "iid random profile-score rows within each respondent; all sampled designs must be identified"}[method]
    return sealed("conjoint_bootstrap_fit", {
        "estimates": table(estimates, columns=["subject", "target", "kind", "term", "attribute", "level", *MOMENT_COLUMNS]),
        "covariance": table(covariances, columns=["subject", "target", *names]),
        "replicates": table(replicas, columns=["subject", "replicate", *names]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "method": method, "sampling": assumption,
        "source_covariance_method": state["covariance"], "all_replicas_identified": True,
        "failed_replicas": 0, "refit_solver": "scaled full-rank float64 QR",
        "replications": b, "seed": seed, "level": level, "quantile": "linear (type 7)",
        "profile_ids": state["training_profile_ids"], "design_scaled": design.tolist(),
        "target_map_scaled": mapping.tolist(), "targets": metadata, "subjects": provenance,
        "settings": settings}, settings)


def respondent_bootstrap(result, values, labels, procedure, settings, b, seed, level, details):
    state = c.intact(result, "conjoint_fit")
    m, k = values.shape
    rng = random.Random(seed)
    draws = [[rng.randrange(m) for _ in range(m)] for _ in range(b)]
    replicas = torch.stack([values[draw].mean(0) for draw in draws])
    rows, covariance = moments(replicas, values.mean(0), level)
    names = [f"target_{i}" for i in range(k)]
    subjects = [record["subject"] for record in state["subjects"]]
    return sealed(procedure, {
        "estimates": table([[name, label, *row] for name, label, row in zip(names, labels, rows)],
                           columns=["target", "label", *MOMENT_COLUMNS]),
        "covariance": table([[name, *covariance[i].tolist()] for i, name in enumerate(names)],
                            columns=["target", *names]),
        "individual": table([[who, *row] for who, row in zip(subjects, values.tolist())],
                            columns=["subject", *names]),
        "replicates": table([[i, *row] for i, row in enumerate(replicas.tolist())],
                            columns=["replicate", *names]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"],
        "source_covariance_method": state["covariance"],
        "sampling": "independent iid whole respondents; mean of fitted individual targets, not latent utilities or calibrated market shares",
        "subjects": subjects, "draw_subject_positions": draws, "individual_values": values.tolist(),
        "replications": b, "seed": seed, "level": level, "quantile": "linear (type 7)",
        "targets": labels, "settings": settings, **details}, settings)


def group_guard(operation, state, k, n, b, device, weights, max_work):
    m, p = len(state["subjects"]), len(state["terms"])
    if m < 2:
        raise AnalysisError("insufficient_subjects", "Whole-respondent bootstrap needs at least two iid respondents.")
    return c.guard(operation, n, p, m, device=device, weights=weights, max_work=max_work,
                   work=m*n*p*p+b*m*k+b*k*k+k*b*max(1, math.ceil(math.log2(b))),
                   extra_buffers={"whole_subject_draws_and_complete_replicas": 256*b*(m+k),
                                  "complete_target_covariance_and_output": 256*k*k})


@resident_cpu
def conjoint_bootstrap_importance(result, *, replications=499, seed=1729, level=0.95,
                                  max_work=100_000_000, device="cpu", weights=None):
    """Iid whole-respondent percentile intervals for mean individual normalized importance.

    Keeps the finite declared level universe. Rejects any undefined individual
    importance. Targets fitted individual importance, without latent-utility
    deconvolution, within-person refits, survey weights or convenience-sample guarantees.
    """
    state = c.intact(result, "conjoint_fit")
    b, seed, level = options(replications, seed, level)
    labels = state["attribute_order"]
    settings = group_guard("conjoint_bootstrap_importance", state, len(labels),
                           sum(len(v) for v in state["attributes"].values()), b, device, weights, max_work)
    output = conjoint_importance(result, max_work=max_work)
    if not output.attrs["complete_group_average_defined"]:
        raise AnalysisError("undefined_importance", "All respondent importance targets must be defined; no subject is discarded.")
    values = torch.tensor(output["individual"].importance_percent.to_numpy().reshape(-1, len(labels)), dtype=torch.float64)
    return respondent_bootstrap(result, values, labels, "conjoint_bootstrap_importance", settings,
                                b, seed, level, {"level_universe": state["attributes"],
                                                 "target": "mean of individually normalized finite-level importance percentages"})


@resident_cpu
def conjoint_bootstrap_shares(result, plan, *, method="first_choice", temperature=1.0, tie_tolerance=0.0,
                              profile="profile_id", replications=499, seed=1729, level=0.95,
                              max_work=100_000_000, device="cpu", weights=None):
    """Iid respondent bootstrap covariance/percentiles for conditional preference shares.

    Preserves first-choice ties, positive-score BTL or declared-temperature logit.
    Resamples entire fitted respondent vectors, with no within-person refit or
    CBC estimation. Intervals target mean fitted shares under iid sampling and
    are not simultaneous bands or calibrated population market-share intervals.
    """
    state = c.intact(result, "conjoint_fit")
    b, seed, level = options(replications, seed, level)
    data, _ = c.profiles(plan, state["attributes"], profile=profile)
    settings = group_guard("conjoint_bootstrap_shares", state, len(data), len(data), b, device, weights, max_work)
    output = conjoint_simulate(result, data, method=method, temperature=temperature,
                              tie_tolerance=tie_tolerance, profile=profile, max_work=max_work)
    values = torch.tensor(output["individual"].conditional_share.to_numpy().reshape(-1, len(data)), dtype=torch.float64)
    return respondent_bootstrap(result, values, [c.label(v) for v in data[profile]], "conjoint_bootstrap_shares",
                                settings, b, seed, level, {"method": method, "temperature": temperature,
                                                          "tie_tolerance": tie_tolerance,
                                                          "profile_column": profile,
                                                          "profiles": data.astype(object).to_numpy().tolist(),
                                                          "profile_columns": list(data.columns)})
