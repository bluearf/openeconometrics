"""Complete-profile RM frequency/summary adapters to the existing RM kernel.

Counts replicate independent whole original-cell subject vectors. Declared
summaries use unbiased subject covariance, not variances of individual cells
from disconnected samples. The existing Mauchly/GG/HF conventions are retained.
https://support.sas.com/documentation/cdl/en/statug/63962/HTML/default/statug_glm_sect037.htm
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as mc
from openecon.econometrics.postest.index_codec import encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace
from . import common as c
from . import manova_factorial as mf
from . import manova_options as mo
from .rm_anova import _contrasts, _finish

MAX_PROFILES = 10_000
MAX_WITHIN_CELLS = 32
_SCHEMA = "openecon.rm.moments.v1"


def _plan(p, groups=1, *, rows=0, profiles=0):
    if not 2 <= p <= MAX_WITHIN_CELLS or profiles > MAX_PROFILES:
        raise AnalysisError("workspace_limit", "RM moment adapters admit 2..32 within cells and 10000 physical subject profiles.")
    base = mf.plan(p, rows=rows, cells=groups, k=groups)
    return plan_workspace("complete-profile RM moments", {
        "categorical_moment_model": base["estimated_workspace_bytes"],
        "wide_profile_and_transform_copies": profiles*p*96,
        "original_and_transformed_group_moments": groups*(p*p+p)*128,
        "long_profile_validation_and_portable_positions": rows*128,
    }).record()


def _index(index, names, *, between=False):
    if names and len(names) == 1 and not isinstance(index, pd.MultiIndex):
        if index.name != names[0]:
            raise AnalysisError("invalid_spec", "A single-factor summary axis must carry its factor name.")
        index = pd.MultiIndex.from_arrays([index.tolist()], names=names)
    return mf._ordered_index(index, names, minimum_factors=0 if between else 1)


def _state(moments, model, within, levels, cells, *, provenance):
    state = mf.moment_state(moments, model, extra={"within": within, "within_levels": levels,
                            "within_cells": [list(cell) for cell in cells],
                            "typed_within_cells": [encode(tuple(cell)) for cell in cells]})
    state["schema"] = _SCHEMA
    state["provenance"] = provenance
    state["sha256"] = mo._digest({key: value for key, value in state.items() if key != "sha256"})
    return state


def _finish_moments(moments, within, within_levels, within_cells, alpha, *, y, provenance):
    original = mf.MomentModel(moments)
    n, p = original.n, original.m
    _plan(p, len(moments.cells))
    mo._covariance(original.error, "pooled original-cell RM residual SSCP", positive=True, rank_limit=original.df_error)
    if original.df_error < p:
        raise AnalysisError("insufficient_observations", "RM moment fits need residual subject df >= original-cell dimension.")
    transform, columns = _contrasts([len(within_levels[name]) for name in within])
    for indices in columns.values():
        mo._projection(original.error, transform[:, indices])
    # Only the scaled subject-mean column has a common response level. The
    # other columns are mathematically zero-sum orthonormal RM contrasts, as
    # in the existing long kernel; subtract the level before applying them.
    anchored = moments.origin-moments.origin[0]
    origin = anchored@transform
    origin[0] += moments.origin[0]*math.sqrt(p)
    offsets = moments.offsets@transform
    scatters = [transform.T@scatter@transform for scatter in moments.scatters]
    scatters = [(scatter+scatter.T)/2 for scatter in scatters]
    mo._finite(origin, offsets, *scatters)
    rotated = mf.Moments(moments.factors, moments.levels, moments.cells, moments.names,
                         origin, offsets, scatters, moments.counts, moments.kind, moments.provenance)
    model = mf.MomentModel(rotated)
    # Existing clean() compares effect sums against total long-response SS.
    # Explicitly refuse positive variation which it would erase at this scale.
    centred = moments.offsets+anchored
    count = torch.tensor(moments.counts, dtype=torch.float64)
    grand = (count[:, None]*centred).sum()/float(n*p)
    total_ss = float(torch.stack(moments.scatters).sum(0).diagonal().sum()
                     +(count[:, None]*(centred-grand).square()).sum())
    mo._finite(torch.tensor(total_ss, dtype=torch.float64))
    for indices in columns.values():
        sse = float(model.error[indices][:, indices].trace())
        if sse > 0 and sse <= 1e-24*total_ss:
            raise AnalysisError("unresolved_precision", "The existing RM ANOVA scale convention cannot resolve positive subject-by-effect error; rescale/refit a stable response basis.")
        for term in model.terms:
            h = float(model.hypothesis(term)[indices][:, indices].trace())
            if h > 0 and h <= 1e-24*total_ss:
                raise AnalysisError("unresolved_precision", "The existing RM ANOVA scale convention cannot resolve this positive effect SS.")
    described = []
    means = moments.offsets+moments.origin
    for g, group in enumerate(moments.cells):
        covariance = moments.scatters[g]/(moments.counts[g]-1) if moments.counts[g] > 1 else None
        for j, cell in enumerate(within_cells):
            described.append([*group, *cell, moments.counts[g], float(means[g, j]),
                              None if covariance is None else float(covariance[j, j].clamp_min(0).sqrt())])
    result = _finish(model, y, within, moments.factors, p, columns, n, total_ss, described,
                     n*p, provenance["n_missing"], alpha, contrast_geometry=(transform, within_levels))
    # Full original-cell geometry/covariance stays separate from the existing
    # univariate RM reporting conventions, using flat portable table axes.
    for key, frame in mf._tables(moments, original, alpha).items():
        result[f"original_{key}"] = frame
    state = _state(moments, original, within, within_levels, within_cells, provenance=provenance)
    result.attrs.update({"procedure": "rm_anova_fweight" if moments.kind == "frequency" else "rm_anova_summary",
                         "rm_moment_state": state, "sample": provenance, "precision": "float64", "device": "cpu",
                         "resource_plan": provenance["resource_plan"], "pointwise_intervals": True,
                         "familywise_intervals": False, "original_covariance_order": "row-major: between-design column then original within cell",
                         "estimated_portable_bytes": original.portable_bytes, "portable_export_limit_bytes": 32*1024**2,
                         "inference": "fixed complete crossed within/between design; independent Gaussian subject vectors with common unrestricted within-subject covariance; counts denote independent original subjects"})
    return saved_summary(result)


@resident_cpu
def rm_anova_fweight(data: Any, y: str, subject: str, within, *, weights, between=None,
                     weight_type="fweight", missing="raise", alpha=0.05):
    """RM tests with exact counts constant over each complete subject profile.

    missing='drop_subject' drops a whole incomplete subject, never individual
    cells. Duplicate cells, changing between factors and rowwise weights are
    refused. Zero-frequency complete profiles are validated then excluded.
    Counts represent independent original subject vectors, not repeated copies
    used to amplify a dependent subject. No Dataset/survey/analytic weights.
    """
    y, subject, weights = (c.check_name(value, name) for value, name in
                           ((y, "y"), (subject, "subject"), (weights, "weights")))
    within = c.name_list(within, "within")
    between = c.name_list(between, "between", minimum=0)
    alpha = c.check_alpha(alpha)
    c.check_choice(weight_type, "weight_type", ("fweight",))
    c.check_choice(missing, "missing", ("raise", "drop_subject"))
    roles = [y, subject, weights, *within, *between]
    if len(set(roles)) != len(roles) or len(within) > mf.MAX_FACTORS or len(between) > mf.MAX_FACTORS:
        raise AnalysisError("invalid_spec", "RM roles must be distinct; at most four within and four between factors are admitted.")
    mf.metadata(roles)
    mf.factor_names([*within, *between])
    frame, _, _, rows = mf._resident(data, roles, [y, weights], "raise" if missing == "raise" else "drop", p=1, complete=False)
    if frame[subject].isna().any():
        raise AnalysisError("invalid_subject", "Missing subject IDs cannot identify whole-profile deletion.")
    within_levels = {name: mf._levels(frame.loc[frame[name].notna(), name]) for name in within}
    p = math.prod(len(values) for values in within_levels.values())
    cells = list(itertools.product(*(within_levels[name] for name in within)))
    mf.metadata(within, within_levels, cells)
    _plan(p, rows=rows)
    cell_lookup = {encode(tuple(cell)): i for i, cell in enumerate(cells)}
    subjects, positions = {}, {}
    for i, value in enumerate(frame[subject].array):
        value = mf._scalar(value)
        key = encode(value)
        if key not in subjects:
            if len(subjects) >= MAX_PROFILES:
                raise AnalysisError("workspace_limit", "RM frequency adapters admit at most 10000 physical subject profiles.")
            subjects[key], positions[key] = value, []
        positions[key].append(i)
    if len(set(subjects.values())) != len(subjects) or len(set(map(str, subjects.values()))) != len(subjects):
        raise AnalysisError("invalid_subject", "Subject IDs contain numeric/text aliases.")
    _plan(p, rows=rows, profiles=len(subjects))
    if sum(len(key) for key in subjects) > 1024**2:
        raise AnalysisError("workspace_limit", "Subject identity metadata exceeds one MiB.")
    profiles, frequencies, groups, kept_ids, kept_positions, dropped_ids = [], [], [], [], [], []
    for key, indices in positions.items():
        block = frame.iloc[indices]
        nonmissing_weight = block.loc[block[weights].notna(), weights]
        # Validate individual counts, then check the subject total once (not the
        # sum of its identical repeated-cell weights).
        values = mf.frequencies(nonmissing_weight, check_total=False)
        if len(set(values)) > 1:
            raise AnalysisError("invalid_weights", "One frequency must be constant across every cell of a subject profile.")
        for name in between:
            labels = [mf._scalar(value) for value in block.loc[block[name].notna(), name]]
            if len(set(encode(value) for value in labels)) > 1:
                raise AnalysisError("invalid_design", "Between-subject factors must be constant within profiles.")
        ordered = {}
        for i in indices:
            row = frame.iloc[i]
            if row[within].isna().any():
                continue
            # Preserve each column's identity dtype; an all-numeric row Series
            # can silently coerce integer factor labels to outcome floats.
            cell = tuple(mf._scalar(frame[name].iloc[i]) for name in within)
            code = cell_lookup[encode(cell)]
            if code in ordered:
                raise AnalysisError("unbalanced_design", "Duplicate subject-cell observations are not frequency weights.")
            ordered[code] = i
        incomplete = len(indices) != p or set(ordered) != set(range(p)) or block[roles].isna().any().any()
        if incomplete:
            if missing == "raise":
                raise AnalysisError("unbalanced_design", "Every subject needs exactly one complete observation in every within cell.")
            dropped_ids.append(key)
            continue
        weight = values[0]
        vector = mc.matrix(frame.iloc[[ordered[j] for j in range(p)]], [y])[:, 0]
        # Validation includes complete zero-frequency profiles.
        profiles.append(vector)
        frequencies.append(weight)
        groups.append([mf._scalar(block[name].iloc[0]) for name in between])
        kept_ids.append(key)
        kept_positions.append([ordered[j] for j in range(p)])
    positive = [i for i, count in enumerate(frequencies) if count > 0]
    if not positive:
        raise AnalysisError("empty_sample", "No positive-frequency complete subject profiles remain.")
    if sum(frequencies) > mo.MAX_COUNT:
        raise AnalysisError("invalid_weights", "Expanded independent subject count exceeds 2**53.")
    group_frame = pd.DataFrame([groups[i] for i in positive], columns=between)
    levels, group_cells, codes = mf.cells_from_frame(group_frame, between)
    resource_plan = _plan(p, len(group_cells), rows=rows, profiles=len(subjects))
    raw = torch.stack([profiles[i] for i in positive])
    frequency = [frequencies[i] for i in positive]
    origin, offsets, scatters, counts = mf.reduce_rows(raw, frequency, codes, group_cells)
    provenance = {"n_missing": sum(len(positions[key]) for key in dropped_ids), "n_input_rows": rows,
                  "physical_subjects": len(subjects), "positive_subjects": len(positive),
                  "n_dropped_subjects": len(dropped_ids), "n_zero_weight_subjects": len(profiles)-len(positive),
                  "expanded_subjects": sum(frequency), "weights": weights, "weight_type": "fweight",
                  "subject": subject, "outcome": y, "subject_order": list(subjects), "dropped_subjects": dropped_ids,
                  "complete_subject_order": kept_ids, "profile_positions": kept_positions,
                  "frequencies": frequencies, "positive_complete_positions": positive, "group_codes": codes,
                  "missing": missing, "sampling_unit": "independent whole original-cell subject profile",
                  "resource_plan": resource_plan}
    names = [f"cell[{i+1}]" for i in range(p)]
    moments = mf.Moments(between, levels, group_cells, names, origin, offsets, scatters, counts, "frequency", provenance)
    return _finish_moments(moments, within, within_levels, cells, alpha, y=y, provenance=provenance)


@resident_cpu
def rm_anova_summary(cell_means: pd.DataFrame, cell_covariances, counts, *, within, between=None, alpha=0.05):
    """RM ANOVA from group original-cell means, subject covariances and counts.

    Columns are the ordered named within-cell MultiIndex; a single within factor
    may use a named Index. Rows similarly name between groups, or one explicit
    group without between factors. Integer counts >=2 and unbiased count-1
    covariances describe independent complete subjects. No subject rows or
    missing-data pattern are synthesized; unresolved projections are refused.
    """
    alpha = c.check_alpha(alpha)
    within, between = c.name_list(within, "within"), c.name_list(between, "between", minimum=0)
    if not isinstance(cell_means, pd.DataFrame) or cell_means.columns.has_duplicates \
            or len(within) > mf.MAX_FACTORS or len(between) > mf.MAX_FACTORS or set(within)&set(between):
        raise AnalysisError("invalid_spec", "Supply distinct bounded within/between roles and a complete labelled cell-mean DataFrame.")
    mf.metadata([*within, *between])
    mf.factor_names([*within, *between])
    within_levels, cells = _index(cell_means.columns, within)
    levels, groups = _index(cell_means.index, between, between=True)
    p = len(cells)
    resource_plan = _plan(p, len(groups))
    origin, offsets, scatters, counts = mf.summary_moments(cell_means, cell_covariances, counts, groups, list(cell_means.columns))
    provenance = {"n_missing": 0, "physical_subjects": None, "expanded_subjects": sum(counts),
                  "sampling_unit": "declared independent complete original-cell subjects",
                  "covariance_divisor": "group subject count-1", "missing": "declared complete summaries",
                  "resource_plan": resource_plan}
    names = [f"cell[{i+1}]" for i in range(p)]
    moments = mf.Moments(between, levels, groups, names, origin, offsets, scatters, counts, "summary", provenance)
    return _finish_moments(moments, within, within_levels, cells, alpha, y="declared original-cell response", provenance=provenance)


def validate_rm_result(result, geometry):
    """Check full original-cell sample moments against saved RM geometry."""
    try:
        state = result.attrs["rm_moment_state"]
        moments, design, names = mf.validate_moments(state, rm=True)
        extra = state["extra"]
        if not isinstance(extra, dict) or set(extra) != {"within", "within_levels", "within_cells", "typed_within_cells"}:
            raise ValueError("RM cell fields")
        within = c.name_list(extra["within"], "saved within")
        levels = extra["within_levels"]
        if len(within) > mf.MAX_FACTORS or not isinstance(levels, dict) or set(levels) != set(within):
            raise ValueError("within metadata")
        for name in within:
            if levels[name] != mf._levels(levels[name]):
                raise ValueError("within level identities")
        cells = [list(cell) for cell in itertools.product(*(levels[name] for name in within))]
        p, k, n = len(cells), len(names), sum(moments.counts)
        _plan(p, len(moments.cells))
        if moments.names != [f"cell[{i+1}]" for i in range(p)] \
                or geometry["cell_order"] != cells or extra["within_cells"] != cells \
                or extra["typed_within_cells"] != [encode(tuple(cell)) for cell in cells] \
                or geometry["within"] != within or geometry["between_design_columns"] != names \
                or geometry["between_levels"] != moments.levels \
                or result.attrs.get("between", []) != moments.factors \
                or geometry["n_subjects"] != n or geometry["df_resid"] != n-k \
                or result.attrs["n_missing"] != moments.provenance["n_missing"]:
            raise ValueError("RM sample, df or geometry order")
        bread = mo._matrix(geometry["bread"], (k, k), "saved RM bread")
        beta = mo._matrix(geometry["coefficients"], (k, p), "saved original-cell RM coefficients")
        covariance = mo._matrix(geometry["residual_cell_covariance"], (p, p), "saved RM covariance")
        count = torch.tensor(moments.counts, dtype=torch.float64)
        normal = design.T@(count[:, None]*design)
        if not torch.allclose(normal@bread, torch.eye(k, dtype=torch.float64), rtol=1e-8, atol=1e-8):
            raise ValueError("RM weighted bread")
        offsets = beta.clone()
        offsets[0] -= moments.origin
        residual = moments.offsets-design@offsets
        condition = design.T@(count[:, None]*residual)
        # Restoring a large common level rounded the legacy cell coefficients.
        # Include that representation error here; requested hypotheses still
        # receive rm_mtest's stricter target-specific precision refusal.
        absolute = design.abs().T@(count[:, None]*(moments.offsets.abs()+design.abs()@beta.abs()+moments.origin.abs()))
        if bool((condition.abs() > 32768*torch.finfo(torch.float64).eps*absolute.clamp_min(1e-300)).any()):
            raise ValueError("RM coefficient normal equations")
        expected = torch.stack(moments.scatters).sum(0)/(n-k)
        scale = expected.diagonal().clamp_min(1e-300).sqrt()
        if not torch.allclose(covariance/torch.outer(scale, scale), expected/torch.outer(scale, scale), rtol=1e-10, atol=1e-10):
            raise ValueError("RM residual covariance/moments")
    except (TypeError, KeyError, ValueError, AttributeError, IndexError):
        raise AnalysisError("invalid_state", "Saved RM moments and original-cell geometry disagree.") from None
