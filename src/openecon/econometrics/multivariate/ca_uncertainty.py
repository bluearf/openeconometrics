"""Full-refit CA bootstrap under two explicitly different count sampling laws.

CA normalization: Nenadic and Greenacre (2007), JSS 20(3),
https://www.jstatsoft.org/article/view/v020i03 . Count sampling models:
https://online.stat.psu.edu/stat504/Lesson03 (section 3.2).

This estimates the smooth, fixed, signed CA functional by refitting each
sampled table. It is not the supplementary-profile projection construction
used for some CA confidence ellipses, nor a bootstrap independence test.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace

from . import common as c

MAX_CATEGORIES = 16
MAX_DIMENSIONS = 4
MAX_TOTAL = 100_000
MAX_REPLICATIONS = 1_999
MAX_PARAMETERS = 128
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024**2
MAX_LABEL_BYTES = 4_096
MAX_IDENTITY_BYTES = 65_536
MAX_IDENTITY_NODES = 10_000
MAX_IDENTITY_DEPTH = 16
SPECTRAL_TOLERANCE = 1e-8
ANCHOR_TOLERANCE = 1e-8
MIN_DIRECTION_COSINE = math.sqrt(.5)


def _codes(values):
    """Bound identity traversal before the lossless shared scalar codec."""
    codes, used_bytes, nodes = [], 0, 0
    for value in values:
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > MAX_IDENTITY_DEPTH or nodes > MAX_IDENTITY_NODES:
                raise AnalysisError("resource_limit", "CA category identities exceed bounded traversal depth/work.")
            if isinstance(item, (str, bytes)) and len(item) > MAX_LABEL_BYTES:
                raise AnalysisError("resource_limit", "One CA category identity exceeds 4096 bytes.")
            if isinstance(item, int) and item.bit_length() > 3*MAX_LABEL_BYTES:
                raise AnalysisError("resource_limit", "One CA integer identity exceeds the portable label bound.")
            if isinstance(item, Decimal) and item.__sizeof__() > MAX_LABEL_BYTES:
                raise AnalysisError("resource_limit", "One CA decimal identity exceeds bounded scalar storage before encoding.")
            if isinstance(item, tuple):
                if len(item) > MAX_IDENTITY_NODES-nodes:
                    raise AnalysisError("resource_limit", "CA tuple identities exceed bounded traversal work.")
                pending.extend((child, depth+1) for child in item)
        try:
            code = encode(value)
        except (AnalysisError, ValueError, TypeError, OverflowError, RecursionError):
            raise AnalysisError("invalid_label", "CA identities need supported lossless scalar/tuple labels.") from None
        size = len(code.encode())
        used_bytes += size
        if size > MAX_LABEL_BYTES or used_bytes > MAX_IDENTITY_BYTES:
            raise AnalysisError("resource_limit", "CA encoded identities exceed the per-label or 64 KiB total bound.")
        codes.append(code)
    return codes


def _input(counts, dimensions):
    if not isinstance(counts, pd.DataFrame):
        raise AnalysisError("unsupported_input", "CA bootstrap requires one labelled resident count DataFrame.")
    r, s = counts.shape
    if not 2 <= r <= MAX_CATEGORIES or not 2 <= s <= MAX_CATEGORIES:
        raise AnalysisError("workspace_limit", "CA bootstrap admits 2 through 16 row and column categories.")
    if dimensions > min(r, s)-1:
        raise AnalysisError("invalid_spec", "Fix a dimension count no larger than min(rows,columns)-1; no clamping is performed.")
    if counts.index.nlevels > MAX_IDENTITY_DEPTH or counts.columns.nlevels > MAX_IDENTITY_DEPTH:
        raise AnalysisError("resource_limit", "CA source axis levels exceed the bounded identity domain.")
    # Encode before Pandas equality/hashing can encounter an unsupported object.
    row_codes = _codes([*counts.index, *counts.index.names])
    column_codes = _codes([*counts.columns, *counts.columns.names])
    if sum(len(code.encode()) for code in row_codes+column_codes) > MAX_IDENTITY_BYTES:
        raise AnalysisError("resource_limit", "Combined CA source identities exceed 64 KiB.")
    if counts.index.has_duplicates or counts.columns.has_duplicates \
            or len(set(row_codes[:r])) != r or len(set(column_codes[:s])) != s:
        raise AnalysisError("duplicate_labels", "CA categories must have distinct unambiguous row and column identities.")
    raw, total = [], 0
    for i in range(r):
        row = []
        for j in range(s):
            value = counts.iat[i, j]
            if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
                raise AnalysisError("invalid_counts", "CA counts must be finite nonnegative exact integers, excluding booleans/complex/coerced strings.")
            try:
                integer = int(value)
            except (ValueError, OverflowError, TypeError):
                raise AnalysisError("invalid_counts", "CA counts must be finite nonnegative exact integers.") from None
            if value != integer or integer < 0:
                raise AnalysisError("invalid_counts", "CA counts must be nonnegative exact integers; no rounding is applied.")
            total += integer
            if total > MAX_TOTAL:
                raise AnalysisError("work_limit", "CA bootstrap admits at most 100000 total sampled counts before any sampling allocation.")
            row.append(integer)
        raw.append(row)
    if total < 20:
        raise AnalysisError("insufficient_observations", "CA bootstrap needs at least 20 total integer observations.")
    return raw, total, row_codes[:r], column_codes[:s], row_codes[r:], column_codes[s:]


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise AnalysisError("numerical_failure", "CA arithmetic exceeds finite float64 precision.")


def _fit(counts, dimensions, anchors, point_axes=None):
    values = counts.to(dtype=c.FLOAT)
    total = counts.sum()
    probability = values/total
    # Exact integer margins preserve genuinely fixed row masses instead of
    # manufacturing roundoff variance by summing separately rounded cells.
    row_mass = counts.sum(1).to(c.FLOAT)/total
    column_mass = counts.sum(0).to(c.FLOAT)/total
    if bool((row_mass <= 0).any()) or bool((column_mass <= 0).any()):
        raise AnalysisError("zero_margin", "Every declared category must have positive mass in every fit; no category is removed.")
    residual = (probability-torch.outer(row_mass, column_mass))/torch.outer(row_mass.sqrt(), column_mass.sqrt())
    left, spectrum, right_t = torch.linalg.svd(residual, full_matrices=False)
    _finite(probability, residual, left, spectrum, right_t)
    scale = float(spectrum[0])
    if scale <= SPECTRAL_TOLERANCE or float(spectrum[dimensions-1]) <= SPECTRAL_TOLERANCE*scale:
        raise AnalysisError("unidentified_axis", "Retained CA singular values must be positive away from numerical zero.")
    gaps = []
    for j in range(dimensions):
        adjacent = []
        if j:
            adjacent.append(spectrum[j-1]-spectrum[j])
        if j+1 < len(spectrum):
            adjacent.append(spectrum[j]-spectrum[j+1])
        gap = float(torch.stack(adjacent).min())/scale
        if gap <= SPECTRAL_TOLERANCE:
            raise AnalysisError("unidentified_axis", "Each retained CA singular value must be simple, including the retained/discarded boundary.")
        gaps.append(gap)
    left, right = left[:, :dimensions], right_t.T[:, :dimensions]
    fixed = [int(left[:, j].abs().argmax()) for j in range(dimensions)] if anchors is None else list(anchors)
    anchor_values = torch.tensor([float(left[i, j]) for j, i in enumerate(fixed)], dtype=c.FLOAT)
    if bool((anchor_values.abs() <= ANCHOR_TOLERANCE).any()):
        raise AnalysisError("unidentified_anchor", "Each fixed row sign anchor must remain away from zero.")
    signs = anchor_values.sign()
    left, right = left*signs, right*signs
    cosines = torch.ones((2, dimensions), dtype=c.FLOAT)
    if point_axes is not None:
        cosines = torch.stack(((point_axes[0]*left).sum(0), (point_axes[1]*right).sum(0)))
        if bool((cosines <= MIN_DIRECTION_COSINE).any()):
            raise AnalysisError("axis_crossing", "A signed row/column axis left its 45-degree point chart; no permutation or Procrustes alignment is applied.")
    rho = spectrum[:dimensions]
    row_standard, column_standard = left/row_mass.sqrt()[:, None], right/column_mass.sqrt()[:, None]
    row_principal, column_principal = row_standard*rho, column_standard*rho
    total_inertia = residual.square().sum()
    vector = torch.cat((total_inertia.reshape(1), row_mass, column_mass,
                        *[torch.cat((rho[j:j+1], rho[j:j+1].square(), row_standard[:, j],
                                     row_principal[:, j], column_standard[:, j], column_principal[:, j]))
                          for j in range(dimensions)]))
    diagnostics = torch.tensor([min(gaps), float(anchor_values.abs().min()), float(cosines.min()),
                                float(rho.min()), float(row_mass.min()), float(column_mass.min())], dtype=c.FLOAT)
    _finite(vector, diagnostics)
    return {"vector": vector, "anchors": fixed, "axes": (left, right), "spectrum": spectrum,
            "probability": probability, "residual": residual, "row_mass": row_mass,
            "column_mass": column_mass, "row_standard": row_standard, "column_standard": column_standard,
            "row_principal": row_principal, "column_principal": column_principal, "diagnostics": diagnostics}


def _draw(probabilities, row_totals, total, sampling, generator):
    r, s = probabilities.shape
    if sampling == "multinomial":
        selected = torch.multinomial(probabilities.flatten(), total, replacement=True, generator=generator)
        return torch.bincount(selected, minlength=r*s).reshape(r, s)
    return torch.stack([torch.bincount(torch.multinomial(probabilities[i], row_totals[i],
                        replacement=True, generator=generator), minlength=s) for i in range(r)])


@c.procedure
@resident_cpu
def ca_bootstrap(counts: pd.DataFrame, *, dimensions: int,
                 sampling: str = "multinomial", anchors: list[Any] | None = None,
                 replications: int = 199, confidence: float = .95, seed: int = 0) -> TableSet:
    """Full signed CA-functional uncertainty under a declared count sampling law.

    ``multinomial`` samples N independent categorical pairs with fitted joint
    cell probabilities and fixed grand total. ``row_multinomial`` independently
    samples each fixed observed row total with its fitted conditional column
    probabilities; row masses remain fixed. The caller must justify that law.
    These are plug-in multinomial bootstrap functionals, not survey, fractional
    weight, selected-dimension or independence-null inference.

    Each requested draw refits both masses and the SVD. Dimensions, descending
    root order and recorded row sign anchors are fixed; all retained roots and
    both signed axis charts must remain resolved. Any failed draw refuses the
    whole result. Population positive margins, simple separated retained roots
    and nonzero anchors are assumed, not certified by these sample gates.
    Percentile intervals are approximate marginal intervals, with no p/df,
    exact coverage, familywise guarantee or supplementary-ellipse equivalence.
    Full joint vectors, sampled count tables, typed category identities and
    RNG/settings persist. Categories and dimensions are never silently removed.
    """
    dimensions = c.check_count(dimensions, "dimensions", maximum=MAX_DIMENSIONS)
    sampling = c.check_choice(sampling, "sampling", ("multinomial", "row_multinomial"))
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1 or (replications+1)*(1-confidence)/2 < 1-1e-12:
        raise AnalysisError("insufficient_replications", "Each percentile tail needs at least one expected order statistic; increase replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63-1)
    raw, total, row_codes, column_codes, row_names, column_names = _input(counts, dimensions)
    r, s = counts.shape
    q = 1+r+s+dimensions*(2+2*r+2*s)
    if q > MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "CA bootstrap admits at most 128 full joint parameters.")
    fixed = None
    if anchors is not None:
        if not isinstance(anchors, (list, tuple)) or len(anchors) != dimensions:
            raise AnalysisError("invalid_spec", "anchors must contain one ordered declared row category per axis.")
        anchor_codes = _codes(anchors)
        if any(code not in row_codes for code in anchor_codes):
            raise AnalysisError("invalid_spec", "Every sign anchor must be an exact declared row category identity.")
        fixed = [row_codes.index(code) for code in anchor_codes]
    sampling_cost = total*(1+math.ceil(math.log2(r*s)))
    work = replications*sampling_cost+(replications+1)*64*min(r, s)**2*max(r, s)
    work += 2*replications*q*q+replications*q*math.ceil(math.log2(replications))+MAX_IDENTITY_NODES
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Cumulative CA sampling/SVD/covariance/quantile work exceeds 250 million operations.")
    numeric_cells = replications*(q+r*s+min(r, s)+6)+q*q+15*q+8*r*s+4*(r+s)
    export_bytes = 32*numeric_cells+4*MAX_IDENTITY_BYTES+128*q+128*(replications+r+s)
    if export_bytes > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "Full CA count-draw/vector state exceeds the conservative 32 MiB output admission.")
    plan = plan_workspace("fixed CA multinomial bootstrap", {
        "one_categorical_sample_and_counts": total*16+r*s*64,
        "point_and_refit_spectral_workspace": 64*r*s*8,
        "all_sampled_count_tables": replications*r*s*16,
        "all_joint_vectors_and_diagnostics": replications*(q+min(r, s)+6)*32,
        "joint_covariance_and_quantiles": q*q*64+q*replications*32,
        "portable_result_admission": export_bytes,
    }).record()
    integer_counts = torch.tensor(raw, dtype=torch.int64)
    point = _fit(integer_counts, dimensions, fixed)
    probabilities = point["probability"] if sampling == "multinomial" else integer_counts.to(c.FLOAT)/integer_counts.sum(1)[:, None]
    row_totals = integer_counts.sum(1).tolist()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = torch.empty((replications, q), dtype=c.FLOAT)
    sampled = torch.empty((replications, r, s), dtype=torch.int64)
    diagnostics = torch.empty((replications, 6), dtype=c.FLOAT)
    spectra = torch.empty((replications, min(r, s)), dtype=c.FLOAT)
    failures = []
    for b in range(replications):
        sampled[b] = _draw(probabilities, row_totals, total, sampling, generator)
        try:
            fitted = _fit(sampled[b], dimensions, point["anchors"], point["axes"])
            vectors[b], diagnostics[b], spectra[b] = fitted["vector"], fitted["diagnostics"], fitted["spectrum"]
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": b+1, "code": getattr(error, "code", "numerical_failure"), "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} CA refits failed. No uncertainty result is returned and no draw is dropped.")
        error.failures = failures
        error.replications_attempted = replications
        error.successful_replications = replications-len(failures)
        raise error
    shifts = vectors-vectors[0]
    mean_shift = shifts.mean(0)
    centered = shifts-mean_shift
    covariance = centered.T@centered/(replications-1)
    covariance = (covariance+covariance.T)/2
    if bool(((centered.abs().amax(0) > 0) & (covariance.diagonal() < torch.finfo(c.FLOAT).tiny)).any()):
        raise AnalysisError("numerical_failure", "A nonconstant joint bootstrap variance underflows normal float64 precision.")
    standard_errors = covariance.diagonal().sqrt()
    intervals = torch.quantile(vectors, torch.tensor([(1-confidence)/2, (1+confidence)/2], dtype=c.FLOAT), dim=0, interpolation="linear")
    _finite(covariance, standard_errors, intervals)
    row_ids, column_ids = c.numbered("Row", r), c.numbered("Col", s)
    order = [["total_inertia", 0, "", 0, ""]]
    order += [["mass", 0, "row", i+1, code] for i, code in enumerate(row_codes)]
    order += [["mass", 0, "column", i+1, code] for i, code in enumerate(column_codes)]
    for j in range(dimensions):
        order += [[kind, j+1, "", 0, ""] for kind in ("rho", "principal_inertia")]
        for axis, codes in (("row", row_codes), ("column", column_codes)):
            for kind in ("standard", "principal"):
                order += [[kind, j+1, axis, i+1, code] for i, code in enumerate(codes)]
    labels = ["total_inertia" if kind == "total_inertia" else
              f"{axis}_mass:{ordinal}" if kind == "mass" else
              f"{kind}:Dim{dimension}" if not axis else
              f"{axis}_{kind}:Dim{dimension}:{ordinal}" for kind, dimension, axis, ordinal, _ in order]
    undefined = torch.full_like(point["vector"], float("nan"))
    coordinate_names = ["mass", *[name for j in range(dimensions) for name in (f"Dim{j+1}_standard", f"Dim{j+1}_principal")]]
    tables = {
        "estimates": c.frame(torch.stack((point["vector"], standard_errors, intervals[0], intervals[1], (vectors[0]-point["vector"])+mean_shift, undefined, undefined), dim=1), columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(vectors, columns=labels, index=range(1, replications+1)),
        "resample_counts": table(sampled.reshape(replications, r*s).numpy(), columns=[f"{a}:{b}" for a in row_ids for b in column_ids], index=range(1, replications+1)),
        "replicate_spectrum": c.frame(spectra, columns=c.numbered("Dim", min(r, s)), index=range(1, replications+1)),
        "replicate_diagnostics": c.frame(diagnostics, columns=["minimum_relative_singular_gap", "minimum_absolute_anchor", "minimum_signed_direction_cosine", "minimum_retained_rho", "minimum_row_mass", "minimum_column_mass"], index=range(1, replications+1)),
        "point_counts": table(integer_counts.numpy(), columns=column_ids, index=row_ids),
        "point_probabilities": c.frame(point["probability"], columns=column_ids, index=row_ids),
        "point_residuals": c.frame(point["residual"], columns=column_ids, index=row_ids),
        "point_spectrum": c.frame(point["spectrum"][:, None], columns=["rho"], index=c.numbered("Dim", min(r, s))),
        "rows": c.frame(torch.stack((point["row_mass"], *[part[:, j] for j in range(dimensions) for part in (point["row_standard"], point["row_principal"])]), dim=1), columns=coordinate_names, index=row_ids),
        "columns": c.frame(torch.stack((point["column_mass"], *[part[:, j] for j in range(dimensions) for part in (point["column_standard"], point["column_principal"])]), dim=1), columns=coordinate_names, index=column_ids),
        "sampling_probabilities": c.frame(probabilities, columns=column_ids, index=row_ids),
        "parameter_order": table(order, columns=["kind", "dimension", "axis", "ordinal", "label_code"], index=labels),
        "category_identities": table([[axis, i+1, code] for axis, codes in (("row", row_codes), ("column", column_codes)) for i, code in enumerate(codes)], columns=["axis", "ordinal", "label_code"]),
    }
    identity = {"row_codes": row_codes, "column_codes": column_codes, "row_names": row_names,
                "column_names": column_names, "row_nlevels": counts.index.nlevels,
                "column_nlevels": counts.columns.nlevels, "counts": raw}
    output = TableSet(tables, title=f"Fixed correspondence-analysis {sampling} bootstrap",
        procedure="ca_bootstrap", sampling=sampling, dimensions=dimensions, n=total,
        row_categories=r, column_categories=s, n_missing=0, missing_policy="no missing count cells or row/category deletion",
        row_label_codes=row_codes, column_label_codes=column_codes,
        source_row_axis_names=row_names, source_column_axis_names=column_names,
        source_row_nlevels=counts.index.nlevels, source_column_nlevels=counts.columns.nlevels,
        source_content_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        sign_anchor_positions=point["anchors"], sign_anchor_codes=[row_codes[i] for i in point["anchors"]],
        anchor_selection="strongest point row singular-vector entry" if anchors is None else "caller declared",
        replications=replications, successful_replications=replications, failed_replications=[], confidence=confidence, seed=seed,
        parameter_order=order, parameter_labels=labels, parameter_dimension=q,
        covariance_divisor=replications-1, quantile_interpolation="linear",
        point_diagnostics=point["diagnostics"].tolist(), spectral_tolerance=SPECTRAL_TOLERANCE,
        anchor_tolerance=ANCHOR_TOLERANCE, minimum_direction_cosine=MIN_DIRECTION_COSINE,
        axis_identity="fixed descending singular-root order and row-anchor signs; both row and column singular axes stay in signed 45-degree point charts; no permutation or alignment",
        covariance_order="total inertia; row masses; column masses; per retained dimension: rho, inertia, row standard/principal, column standard/principal",
        resample_counts_shape=[replications, r, s], resample_counts_order="replication, source row ordinal, source column ordinal",
        fixed_grand_total=True, fixed_row_totals=sampling == "row_multinomial", row_totals=row_totals,
        rng="torch.Generator CPU; torch.multinomial(replacement=True), then bincount; increasing replicate then source-row order", rng_version=torch.__version__,
        reestimate_masses_and_axes_every_draw=True, p_values_available=False, inference_df_available=False,
        precision="float64", device="cpu", state_schema="openecon.ca_bootstrap.v1",
        inference_target="fixed simple signed CA functional of joint cell probabilities" if sampling == "multinomial" else "fixed simple signed CA functional of independent row-conditional probabilities at the declared row allocation",
        inferential_assumptions="independent categorical counts under the declared fixed-total sampling law; fixed support/dimension; positive population margins; simple separated positive retained singular values and fixed nonzero anchors; population regularity assumed, not established by sample guards",
        uncertainty="first-order plug-in multinomial marginal percentile bootstrap", familywise_intervals=False,
        resource_plan=plan, estimated_work=work, estimated_complete_export_bytes=export_bytes,
        notes=["No exact coverage, selected-axis inference, survey/fractional counts, independence-test p-value, supplementary-ellipse or licensed vendor parity is claimed.",
               "Every requested draw must pass identical margin/rank/gap/anchor/chart guards; failed draws never produce subset intervals.",
               "Discarded and centering-null roots are saved diagnostics without individual uncertainty intervals."])
    return saved_summary(output)
