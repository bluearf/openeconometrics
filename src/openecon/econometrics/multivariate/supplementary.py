"""Supplementary correspondence profiles on persisted active axes."""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from . import common as c
from .summary import geometry


@c.procedure
def ca_project(result, profiles, *, axis: str = "row"):
    """Project supplementary rows or columns without changing the active CA fit.

    profiles is a table with one row per supplementary point and columns named
    after every active category on the opposite axis, in any order. Counts are
    normalized within each profile. All categories must be present; insert zero
    explicitly for absent categories. No passive mass/contribution is invented.

    Example
    -------
    >>> import openecon as oe
    >>> fit = oe.ca({'r': ['a', 'a', 'b', 'b'], 'c': ['x', 'y', 'x', 'y'],
    ...              'n': [8, 2, 1, 9]}, 'r', 'c', weights='n', dimensions=1)
    >>> ca_project(fit, {'x': [4], 'y': [1]}).shape[0]
    1
    """
    c.check_result(result, "ca", "ca_project")
    c.check_choice(axis, "axis", ("row", "column"))
    state = result.attrs.get("projection_state")
    if not isinstance(state, dict) or state.get("version") != 1:
        raise AnalysisError("invalid_result", "Saved CA result does not contain supplementary projection state.")
    opposite = "column" if axis == "row" else "row"
    labels = state.get(f"{opposite}_labels")
    if not isinstance(labels, list) or len(set(labels)) != len(labels) or len(labels) < 2:
        raise AnalysisError("invalid_result", "Saved opposite-axis category labels are invalid.")
    raw = c.source(profiles)
    if len(raw) > 16384:
        raise AnalysisError("work_limit", "Supplementary projection supports at most 16,384 profiles per call.")
    plan = geometry(len(labels), rows=len(raw))
    if set(raw.columns) != set(labels):
        raise AnalysisError("invalid_spec", "Profile columns must match all active opposite-axis categories exactly.")
    c.require_numeric(raw, labels)
    x = c.matrix(raw, labels)
    if bool((x < 0).any()) or not bool((x.sum(1) > 0).all()) or not bool(torch.isfinite(x.sum(1)).all()):
        raise AnalysisError("invalid_profile", "Profiles must contain finite nonnegative counts with a positive total.")
    profile = x / x.sum(1, keepdim=True)
    try:
        mass = torch.as_tensor(state[f"{opposite}_mass"], dtype=c.FLOAT)
        standard = torch.as_tensor(state[f"{opposite}_standard"], dtype=c.FLOAT)
        singular = torch.as_tensor(state["singular_values"], dtype=c.FLOAT)
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        raise AnalysisError("invalid_result", "Saved supplementary state is incomplete.") from exc
    if mass.shape != (len(labels),) or standard.shape != (len(labels), len(singular)) \
            or not bool(torch.isfinite(mass).all()) or not bool((mass > 0).all()) \
            or not bool(torch.isfinite(standard).all()) or not bool(torch.isfinite(singular).all()) \
            or bool((singular < 0).any()) or abs(float(mass.sum()) - 1) > 1e-10 \
            or len(singular) != result.attrs.get("dimensions"):
        raise AnalysisError("invalid_result", "Saved supplementary state is invalid.")
    principal = profile @ standard
    identified = singular > 1e-12 * singular.max().clamp_min(1e-300)
    principal = torch.where(identified[None, :], principal, torch.zeros_like(principal))
    distance = ((profile - mass).square() / mass).sum(1)
    principal = torch.where((distance < 1e-24)[:, None], torch.zeros_like(principal), principal)
    safe = distance.clamp_min(1e-300)
    values = [distance, (principal.square().sum(1) / safe).clamp(0, 1)]
    names = ["chi2_distance", "quality"]
    for j in range(len(singular)):
        if bool(identified[j]):
            values += [principal[:, j], principal[:, j] / singular[j], principal[:, j].square() / safe]
        else:
            unknown = torch.full_like(distance, float("nan"))
            values += [unknown, unknown, unknown]
        names += [f"dim{j+1}", f"dim{j+1}_standard", f"dim{j+1}_sqcorr"]
    return c.frame(torch.stack(values, dim=1), columns=names, index=raw.index,
                   procedure="ca_project", axis=axis, supplementary=True,
                   n=len(raw), active_total=result.attrs["n"], dimensions=len(singular),
                   resource_plan=plan, inference="descriptive; no sampling CI", precision="float64",
                   notes=["Active axes and inertia are fixed. Near-zero axes are unidentified: their coordinates and squared correlations are undefined."])
