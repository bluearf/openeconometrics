"""Public bounded exact conditional regression and saved-distribution inference."""

from __future__ import annotations
from openecon.econometrics.resident_cpu import resident_cpu
from . import engine


@resident_cpu
def exact_logit_fit(
    data,
    y: str,
    x: str,
    *,
    nuisance=None,
    missing="raise",
    max_states=20000,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Single-target conditional Bernoulli CMLE after conditioning intercept and up to 3 nuisance statistics.

    At most 16 original rows; y is numeric 0/1, x/nuisance integers +/-8.
    Full enumeration, saved allocations, exact constraints and honest infinite
    boundaries. No nuisance estimates, Wald inference, MUE or weighted support.
    """
    return engine.fit(
        "logit", data, y, x, nuisance, None, missing, max_states, max_work, device, weights
    )


@resident_cpu
def exact_logit_ci(result, *, level=0.95, max_work=100_000_000):
    """Invert inclusive conditional Bernoulli statistic tails for an exact central coefficient interval.

    Reads saved full fit state; explicit infinite endpoints and root traces.
    No mid-p correction or normal/Wald interval. Finite roots require [-40,40].
    """
    return engine.intervals(result, "logit", level, max_work)


@resident_cpu
def exact_logit_test(result, *, null=0.0, max_work=100_000_000):
    """Two-sided probability-ordered grouped-statistic conditional Bernoulli test at finite null beta.

    Includes ties using log-probability tolerance 1e-12; records complete null
    distribution/critical region. No score/Wald or individual-allocation ordering.
    """
    return engine.test(result, "logit", null, max_work)


@resident_cpu
def exact_logit_moments(result, *, coefficient=None, max_work=100_000_000):
    """Conditional Bernoulli training response means/full covariance at known beta or labelled CMLE plugin.

    Preserves original positions/missing rows. Boundary plugin uses the complete
    limiting support face; conditional_response_sd is response spread, not an
    estimated-mean SE. No unconditional new-row prediction or parameter CI.
    """
    return engine.moments(result, "logit", coefficient, max_work)


@resident_cpu
def exact_poisson_fit(
    data,
    y: str,
    x: str,
    *,
    nuisance=None,
    exposure=None,
    missing="raise",
    max_states=20000,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Single-target conditional Poisson CMLE with known exposure and conditioned nuisance statistics.

    At most 8 original rows/24 total events/20000 pre-filter compositions; x and
    up to 3 nuisance columns integers +/-8. Known exposure column positive in
    [1e-12,1e12], default unity. Retains factorial/exposure allocation weights
    and honest infinite CMLE. No nuisance estimates or observation weights.
    """
    return engine.fit(
        "poisson", data, y, x, nuisance, exposure, missing, max_states, max_work, device, weights
    )


@resident_cpu
def exact_poisson_ci(result, *, level=0.95, max_work=100_000_000):
    """Invert inclusive exposure-aware conditional Poisson tails for an exact central coefficient interval.

    Complete saved finite distribution; explicit infinite endpoints/root traces.
    Finite roots require [-40,40]; no mid-p/Wald or unconditioned rate interval.
    """
    return engine.intervals(result, "poisson", level, max_work)


@resident_cpu
def exact_poisson_test(result, *, null=0.0, max_work=100_000_000):
    """Two-sided probability-ordered grouped-statistic conditional Poisson test at finite null beta.

    Aggregates factorial/exposure weighted allocations before probability
    ordering, includes ties at log tolerance 1e-12 and saves the critical region.
    """
    return engine.test(result, "poisson", null, max_work)


@resident_cpu
def exact_poisson_moments(result, *, coefficient=None, max_work=100_000_000):
    """Conditional count training means/full covariance with known beta or explicitly labelled CMLE plugin.

    Uses actual factorial/exposure allocation probabilities, including limiting
    boundary faces. Preserves sample positions/labels and dropped rows. Response
    SD is not mean estimation SE; no unconditional/new-row/parameter inference.
    """
    return engine.moments(result, "poisson", coefficient, max_work)
