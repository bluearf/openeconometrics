"""Bounded iid interval-censored NPMLE with exact support and KKT checks.

Turnbull (1976) supplies the self-consistency likelihood.  Gentleman and
Geyer (1994), sections 2.2 and 4, justify the independent KKT optimality gate:
small EM changes alone cannot certify a maximum.  Location within a support
interval is left unidentified, including the unbounded right-tail interval.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.multivariate.common import procedure
from .common import MAX_SUPPORT, interval_data, output, prediction_times, workspace


SOURCES = ["https://doi.org/10.1111/j.2517-6161.1976.tb01597.x",
           "https://doi.org/10.1093/biomet/81.3.618"]
WORK_LIMIT = 500_000_000


def _controls(maxiter, tol):
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or not 1 <= maxiter <= 10000:
        raise AnalysisError("invalid_option", "maxiter must be an integer in 1..10000.")
    if isinstance(tol, bool) or not isinstance(tol, Real) or not math.isfinite(tol) or not 1e-12 <= tol <= 1e-3:
        raise AnalysisError("invalid_option", "tol must be a finite number in [1e-12,1e-3].")
    return int(maxiter), float(tol)


def _support(data):
    """Sweep incidence patterns, retaining only maximal intersections.

    Python integer bitsets avoid an n-by-all-endpoints dense allocation.
    Equal incidence patterns are merged and strict subsets are dominated:
    moving their mass to a containing pattern cannot lower any observation
    probability and strictly improves at least one likelihood term.
    Exact observations participate only in their singleton boundary cells.
    """
    lower, upper = data.lo.tolist(), data.hi.tolist()
    events = {}
    for i, (lo, hi, kind) in enumerate(zip(lower, upper, data.kind)):
        bit = 1 << i
        if kind == "exact":
            events.setdefault(lo, [0, 0, 0])[2] |= bit
        else:
            events.setdefault(lo, [0, 0, 0])[0] |= bit
            if math.isfinite(hi):
                events.setdefault(hi, [0, 0, 0])[1] |= bit
    patterns = set()
    active = events.get(0., [0, 0, 0])[0]
    previous = 0.
    for endpoint in sorted(e for e in events if e > 0):
        start, end, exact = events[endpoint]
        if endpoint > previous and active:
            patterns.add(active)
        point = active | exact
        if point:
            patterns.add(point)
        active = (active & ~end) | start
        previous = endpoint
    if active:
        patterns.add(active)
    retained = []
    for pattern in sorted(patterns, key=lambda v: (-v.bit_count(), v)):
        if any(pattern & other == pattern for other in retained):
            continue
        retained.append(pattern)
        if len(retained) > MAX_SUPPORT:
            raise AnalysisError("resource_limit", "The maximal-intersection support exceeds 256 cells; no support thinning is performed.")
    cells = []
    for pattern in retained:
        members = [i for i in range(data.n) if pattern >> i & 1]
        exact = [lower[i] for i in members if data.kind[i] == "exact"]
        if exact:
            lo = hi = exact[0]
            if any(point != lo for point in exact):
                raise AnalysisError("numerical_failure", "Distinct exact observations cannot share a support atom.")
            point = True
        else:
            lo, hi = max(lower[i] for i in members), min(upper[i] for i in members)
            point = False
            if not lo < hi:
                raise AnalysisError("numerical_failure", "A maximal support intersection is empty under the declared open-left convention.")
        cells.append(dict(lower=lo, upper=None if math.isinf(hi) else hi,
                          left_closed=point, right_closed=math.isfinite(hi),
                          kind="exact_point" if point else "right_tail_interval" if math.isinf(hi) else "interval",
                          members=members))
    cells.sort(key=lambda cell: (cell["lower"], math.inf if cell["upper"] is None else cell["upper"]))
    if not cells:
        raise AnalysisError("unidentified_mass", "No nonempty likelihood support could be identified.")
    return cells


def _fit(A, maxiter, tol):
    n, k = A.shape
    mass = torch.full((k,), 1./k, dtype=torch.float64, device="cpu")
    q = A @ mass
    if bool((q <= 0).any()):
        raise AnalysisError("invalid_support", "Every observation must contain at least one retained support cell.")
    likelihood = float(q.log().sum())
    trace = []
    for iteration in range(1, maxiter+1):
        gradient = (A.T @ q.reciprocal()) / n
        updated = mass * gradient
        updated /= updated.sum()
        new_q = A @ updated
        if not bool(torch.isfinite(updated).all()) or not bool(torch.isfinite(new_q).all()) or bool((new_q <= 0).any()):
            raise AnalysisError("numerical_failure", "The interval likelihood became nonfinite or lost an observation's support.")
        new_likelihood = float(new_q.log().sum())
        if new_likelihood < likelihood - 2e-12*n:
            raise AnalysisError("numerical_failure", "An EM update decreased the interval log likelihood beyond the floating-point gate.")
        em_likelihood = new_likelihood
        updated, new_q, new_likelihood, acceleration = _newton(A, updated, new_q, new_likelihood)
        score = (A.T @ new_q.reciprocal()) / n
        # For mean log likelihood, score·mass=1; this Frank--Wolfe dual
        # gap is an independent upper bound on objective suboptimality.
        dual_gap = max(0., float(score.max() - score @ updated))
        step = float((updated-mass).abs().max())
        change = abs(new_likelihood-likelihood) / n
        trace.append(dict(iteration=iteration, log_likelihood=new_likelihood,
                          em_log_likelihood=em_likelihood, acceleration=acceleration,
                          mean_log_likelihood_change=change, maximum_mass_change=step,
                          normalized_dual_gap=dual_gap))
        mass, q, likelihood = updated, new_q, new_likelihood
        if dual_gap <= tol and change <= tol:
            return mass, q, likelihood, score, trace
    raise AnalysisError("nonconvergence", f"Turnbull EM did not pass the likelihood/KKT dual-gap gates within {maxiter} iterations (normalized dual gap={dual_gap:.6g}); no NPMLE is reported.")


def _newton(A, mass, probabilities, likelihood):
    """Optional monotone simplex-tangent Newton acceleration of an EM step.

    Positive-start EM keeps every support cell available.  This Newton step
    also keeps all masses positive (99% of a boundary-reaching step), avoids
    arbitrary zero pruning and uses an Armijo likelihood line search.  A
    bounded/ill-conditioned Newton system simply leaves the valid EM step
    in place, with that numerical strategy reported in the trace; the same
    final KKT certificate remains mandatory for every route.
    """
    n, k = A.shape
    if k == 1:
        return mass, probabilities, likelihood, "not_needed_single_cell"
    if k > 64:
        return mass, probabilities, likelihood, "EM_only_above_64_cell_Newton_budget"
    score = (A.T @ probabilities.reciprocal()) / n
    white = A / probabilities[:, None]
    information = (white.T @ white) / n
    if not bool(torch.isfinite(information).all()) or not float(torch.linalg.cond(information)) < 1e12:
        return mass, probabilities, likelihood, "EM_only_ill_conditioned_Newton_system"
    cholesky, flag = torch.linalg.cholesky_ex(information)
    if int(flag) != 0:
        return mass, probabilities, likelihood, "EM_only_failed_Newton_factorization"
    rhs = torch.stack((score, torch.ones(k, dtype=torch.float64, device="cpu")), dim=1)
    solution = torch.cholesky_solve(rhs, cholesky)
    denominator = float(solution[:, 1].sum())
    if not denominator > 0 or not bool(torch.isfinite(solution).all()):
        return mass, probabilities, likelihood, "EM_only_invalid_Newton_direction"
    direction = solution[:, 0] - (float(solution[:, 0].sum())/denominator) * solution[:, 1]
    slope = float(score @ direction)
    if not math.isfinite(slope) or slope <= 0:
        return mass, probabilities, likelihood, "EM_only_no_Newton_ascent"
    negative = direction < 0
    maximum = float((mass[negative]/(-direction[negative])).min()) if bool(negative.any()) else math.inf
    step = min(1., .99*maximum)
    for _ in range(32):
        candidate = mass+step*direction
        candidate /= candidate.sum()
        q = A@candidate
        if bool(torch.isfinite(candidate).all()) and bool((candidate > 0).all()) and bool((q > 0).all()):
            value = float(q.log().sum())
            if math.isfinite(value) and value >= max(likelihood, likelihood+1e-4*n*step*slope):
                return candidate, q, value, "monotone_tangent_Newton"
        step *= .5
    return mass, probabilities, likelihood, "EM_only_Newton_line_search_refused"


def _bounds(cells, mass, times):
    rows = []
    for time in times.tolist():
        certain, possible, unidentified = [], [], []
        for j, cell in enumerate(cells):
            point = cell["kind"] == "exact_point"
            included = cell["upper"] is not None and time >= cell["upper"]
            could = time >= cell["lower"] if point else time > cell["lower"]
            if included:
                certain.append(j)
            if could:
                possible.append(j)
            if could and not included and float(mass[j]) > 0:
                unidentified.append(j)
        lower = float(mass[certain].sum()) if certain else 0.
        upper = float(mass[possible].sum()) if possible else 0.
        lower, upper = max(0., min(1., lower)), max(0., min(1., upper))
        rows.append(dict(time=time, cdf_lower=lower, cdf_upper=upper,
                         survival_lower=1.-upper, survival_upper=1.-lower,
                         point_identified=not unidentified,
                         unidentified_cells=len(unidentified)))
    return rows


@procedure
@torch.no_grad()
def turnbull(lower, upper, *, times=None, device="cpu", weights=None,
             maxiter=1000, tol=1e-9) -> TableSet:
    """Fit an iid interval-censored NPMLE without assigning invented event times.

    Finite intervals are (lower,upper]; lower=upper>0 is an exact atom,
    lower=0 is left censoring and upper=None/+inf is T>lower right censoring.
    Censoring must be independent/noninformative.  Native CPU float64 EM
    starts with positive support masses, uses safeguarded monotone Newton
    acceleration for <=64 cells, and must pass monotonicity, likelihood and
    normalized KKT dual-gap gates. Full-column-rank membership is required
    as a conservative certificate that support masses are unique.

    Returned intervals/atoms and all observation probabilities are saved.
    Requested-time CDF/survival bounds describe unidentified location within
    fitted support cells, not confidence intervals.  Right-tail mass occupies
    an unbounded finite-event interval rather than a cure atom at infinity.
    No covariance/SE/p-values or truncation/covariate/weighted inference.
    <=4096 observations, <=256 support cells/times, 1..10000 iterations.
    Preflight work estimate n*k²+maxiter*(n*k+(n*k²+k³ if k<=64 else 0))
    includes the once-only rank SVD and must be <=500 million before the
    dense support matrix is allocated.
    """
    maxiter, tol = _controls(maxiter, tol)
    data = interval_data(lower, upper, device=device, weights=weights)
    cells = _support(data)
    if times is None:
        defaults = sorted({cell["upper"] for cell in cells if cell["upper"] is not None})
        times = defaults or [0.]
    requested = prediction_times(times)
    resource = workspace(data.n, len(requested), support=len(cells))
    k = len(cells)
    work_estimate = data.n*k*k+maxiter*(data.n*k+(data.n*k*k+k**3 if k <= 64 else 0))
    if work_estimate > WORK_LIMIT:
        raise AnalysisError("resource_limit", "Turnbull's declared iteration/support work exceeds 500 million estimated units; reduce maxiter or the supported dataset, without thinning support.")
    A = torch.zeros((data.n, len(cells)), dtype=torch.float64, device="cpu")
    for j, cell in enumerate(cells):
        A[cell["members"], j] = 1.
    if bool((A.sum(1) == 0).any()):
        raise AnalysisError("invalid_support", "Some observation has no maximal-intersection support.")
    singular = torch.linalg.svdvals(A)
    if len(singular) < len(cells) or float(singular[-1]) <= float(singular[0])*1e-10:
        raise AnalysisError("unidentified_mass", "The support incidence matrix lacks a numerically full column rank certificate; nonunique mass inference is unsupported.")
    mass, probabilities, likelihood, scores, trace = _fit(A, maxiter, tol)
    support = [{"cell":j+1, **{key:value for key,value in cell.items() if key != "members"},
                "mass":float(mass[j]), "normalized_score":float(scores[j]),
                "observations_containing_cell":len(cell["members"])} for j,cell in enumerate(cells)]
    membership = [[j+1 for j in range(len(cells)) if bool(A[i,j])] for i in range(data.n)]
    state = dict(schema="openecon.turnbull.v1", support=support, mass=mass.tolist(),
                 observation_cell_membership=membership, observation_probabilities=probabilities.tolist(),
                 log_likelihood=likelihood, iterations=len(trace), normalized_dual_gap=trace[-1]["normalized_dual_gap"],
                 mass_sum=float(mass.sum()), support_rank=len(cells), tol=tol, maxiter=maxiter)
    state["checksum"] = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    frames = dict(
        support=table(support),
        observations=table([dict(observation=i+1, lower=data.settings["lower"][i], upper=data.settings["upper"][i],
                                 censoring_kind=data.kind[i], probability=float(probabilities[i]),
                                 log_likelihood=float(probabilities[i].log()),
                                 admitted_cells=json.dumps(membership[i])) for i in range(data.n)]),
        cdf_bounds=table(_bounds(cells, mass, requested)),
        convergence=table(trace),
        fit=table([dict(n=data.n, support_cells=len(cells), support_rank=len(cells), log_likelihood=likelihood,
                        iterations=len(trace), normalized_dual_gap=trace[-1]["normalized_dual_gap"], mass_sum=float(mass.sum()))]),
    )
    settings = dict(data.settings, **resource, times=requested.tolist(), maxiter=maxiter, tol=tol,
                    work_estimate=work_estimate, work_limit=WORK_LIMIT,
                    work_formula="n*support^2+maxiter*(n*support+(n*support^2+support^3 if support<=64 else 0))",
                    log_likelihood=likelihood, iterations=len(trace), convergence="likelihood and KKT dual-gap gates passed",
                    algorithm="positive uniform-start EM on maximal intersections with safeguarded monotone simplex-tangent Newton acceleration for <=64 cells; no mass pruning or midpoint imputation",
                    normalized_dual_gap=trace[-1]["normalized_dual_gap"],
                    total_log_likelihood_suboptimality_bound=data.n*trace[-1]["normalized_dual_gap"],
                    support_rank=len(cells), mass_identification="full column rank certificate for unique support-cell masses; conservative refusal if deficient",
                    location_identification="sharp location bounds conditional on numerical fitted cell masses; locations inside intervals are unidentified",
                    uncertainty="location identification bounds only; sampling covariance/SE/CI/df/p-values not reported",
                    right_tail="mass in (lower,+infinity) over finite event times; no atom at infinity and no estimated cure fraction",
                    npmle_state=state, source=SOURCES)
    return output("turnbull", frames, settings,
                  notes=["Returned cell masses satisfy the declared numerical likelihood/KKT tolerance; EM changes alone are not treated as an optimality certificate.",
                         "The CDF/survival envelope reflects unidentified event locations inside support intervals, not sampling confidence or a chosen midpoint distribution.",
                         "Exact observations are singleton atoms. A right-tail cell contains possible future finite event times; its mass is not a cure estimate.",
                         "This complete-data route excludes delayed entry, truncation, covariates, informative censoring and dependent/weighted observations."])
