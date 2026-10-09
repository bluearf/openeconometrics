"""Paired hidden-bias bounds and design-based sharp-null randomization.

Rosenbaum's paired assignment-odds model bounds each positive-sign probability
between 1/(1+Gamma) and Gamma/(1+Gamma). The monotone sign statistic therefore
has exact binomial tail bounds. This is the paired *sign* test, not a signed-rank
procedure, an effect interval or a test that matching removed confounding.
Primary author reference: https://doi.org/10.1093/biomet/74.1.13.

Randomization flips the treatment assignment independently within each pair,
under a declared paired experiment and Fisher's constant additive sharp null.
The distinction from an average-effect null is discussed in the authors' text:
https://rdpackages.github.io/references/Cattaneo-Idrobo-Titiunik_2024_CUP.pdf.
No row shuffling or empirical assignment-design inference is performed.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import beta_inc
from openecon.resources import plan_workspace

from . import common as m


def _pair_key(value):
    """Typed scalar identifiers prevent True/1 or text/numeric conflation."""
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return (type(value).__name__, value), value
    if isinstance(value, float) and math.isfinite(value):
        return ("float", value), value
    raise AnalysisError(
        "invalid_pair",
        "Pair identifiers must be nonmissing finite numeric, boolean or text scalars.",
    )


def _topology(selected, treatment, pair):
    d = m.binary(selected, treatment)
    groups = {}
    for i, label in enumerate(selected[pair]):
        key, safe = _pair_key(label)
        groups.setdefault(key, [safe, []])[1].append(i)
    rows = []
    for key, (label, members) in groups.items():
        if len(members) != 2 or float(d[members].sum()) != 1:
            raise AnalysisError(
                "invalid_pair",
                "Every pair must contain exactly two rows: one numeric treatment=1 and one treatment=0.",
            )
        treated, control = sorted(members, key=lambda i: -float(d[i]))
        rows.append((key, label, treated, control))
    return rows


def _sample(data, y, treatment, pair, missing, max_work):
    names = [
        c.check_name(name, role)
        for name, role in ((y, "y"), (treatment, "treatment"), (pair, "pair"))
    ]
    if len(set(names)) != 3:
        raise AnalysisError(
            "invalid_spec", "Outcome, treatment and pair must name distinct columns."
        )
    # The original assignment topology is validated before outcome deletion.
    # Missing assignment/identity is refused; both missing outcomes can remove
    # a complete pair, but one missing member cannot change its design.
    design, design_metadata = m.sample(
        data, [treatment, pair], numeric=[treatment], missing="raise", max_work=max_work
    )
    original_pairs = _topology(design, treatment, pair)
    selected, metadata = m.sample(
        data, names, numeric=[y, treatment], missing=missing, max_work=max_work, cost=5
    )
    used_pairs = _topology(selected, treatment, pair)
    values = m.tensor(selected, y)
    differences = torch.tensor(
        [float(values[a] - values[b]) for _, _, a, b in used_pairs],
        dtype=torch.float64,
        device="cpu",
    )
    if not bool(torch.isfinite(differences).all()):
        raise AnalysisError("numerical_failure", "Pair contrasts exceed finite float64 range.")
    pair_rows = []
    for _, label, a, b in used_pairs:
        pair_rows.append(
            [
                label,
                metadata["positions"][a],
                metadata["positions"][b],
                float(values[a]),
                float(values[b]),
                float(values[a] - values[b]),
            ]
        )
    metadata.update(
        n_pairs=len(used_pairs),
        n_input_pairs=len(original_pairs),
        n_removed_pairs=len(original_pairs) - len(used_pairs),
        design_sample_sha256=design_metadata["sample_sha256"],
        pair_missing_policy="complete outcome-missing pairs may be removed; broken pairs and missing identities/assignment are refused",
    )
    return differences, pair_rows, metadata


def _tail(k, n, probability):
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # P(Binomial(n,p)>=k)=I_p(k,n-k+1), computing the tail directly.
    return float(beta_inc(k, n - k + 1, probability))


def _pair_table(rows):
    out = m.frame(
        rows,
        columns=[
            "pair",
            "treated_position",
            "control_position",
            "treated_outcome",
            "control_outcome",
            "difference",
        ],
    )
    # Identifiers have typed scalar semantics; mixed floats and large integers
    # must not pass through a float64 inference step in pandas.
    out["pair"] = pd.Series([row[0] for row in rows], dtype=object)
    return out


@m.procedure
def rosenbaum_bounds(
    data: Any,
    y: str,
    treatment: str,
    pair: str,
    *,
    gammas: Sequence[float] = (1.0, 1.5, 2.0),
    alternative: str = "greater",
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Exact one-sided paired sign-test bounds under Rosenbaum hidden bias.

    Each original pair must contain exactly one treated and one control row.
    Positive contrasts favour ``alternative='greater'``; negative contrasts
    favour ``'less'``. Exact zero differences are conditioned out and reported.
    At Gamma=1 both bounds equal the exact paired sign-test p-value. Returned
    bounds concern a sharp no-effect null, not a confidence interval for ATE.
    ``gammas`` is a nonempty ascending list of finite values >=1. Generic
    weights, Dataset replay and non-CPU execution are refused.
    """
    m.options(device, weights, max_work)
    c.check_choice(alternative, "alternative", ("greater", "less"))
    if (
        isinstance(gammas, (str, bytes))
        or not isinstance(gammas, (list, tuple))
        or not 1 <= len(gammas) <= 256
    ):
        raise AnalysisError(
            "invalid_option", "gammas must contain 1..256 ascending finite numbers >=1."
        )
    gamma_values = [c.check_number(value, "gamma", minimum=1) for value in gammas]
    if any(b <= a for a, b in zip(gamma_values, gamma_values[1:])):
        raise AnalysisError(
            "invalid_option", "gammas must be strictly increasing without duplicates."
        )
    differences, pair_rows, metadata = _sample(data, y, treatment, pair, missing, max_work)
    n = int((differences != 0).sum())
    positive = int((differences > 0).sum())
    k = positive if alternative == "greater" else n - positive
    planned_work = metadata["n_input"] * 5 + len(differences) * len(gamma_values) * 8
    m.work(planned_work, max_work, "complete sample/topology and all paired sensitivity bounds")
    bounds = []
    for gamma in gamma_values:
        probability_low = 1.0 / (1.0 + gamma)
        probability_high = gamma / (1.0 + gamma)
        bounds.append(
            [
                gamma,
                _tail(k, n, probability_low),
                _tail(k, n, probability_high),
                probability_low,
                probability_high,
            ]
        )
    settings = dict(
        y=y,
        treatment=treatment,
        pair=pair,
        gammas=gamma_values,
        alternative=alternative,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    state = dict(
        target="one-sided paired sign-test sharp-null p-value bounds",
        pair_rows=pair_rows,
        differences=differences.tolist(),
        positive_signs=positive,
        negative_signs=n - positive,
        informative_pairs=n,
        zero_difference_pairs=len(differences) - n,
        sign_statistic=k,
        bounds=bounds,
        planned_work=planned_work,
        covariance=None,
        standard_error=None,
        df=None,
        confidence_interval=None,
        inference="exact binomial tail bounds; no estimated-effect covariance",
        assumptions=[
            "independent assignment across pairs",
            "within-pair treatment odds ratio bounded by Gamma",
            "sharp no-treatment-effect null",
            "whole-pair outcome availability is fixed independently of treatment assignment",
        ],
        tie_rule="exact zero differences conditioned out; no magnitude ranks",
    )
    return m.result(
        "rosenbaum_bounds",
        {
            "bounds": m.frame(
                bounds,
                columns=["gamma", "p_lower", "p_upper", "probability_lower", "probability_upper"],
            ),
            "pairs": _pair_table(pair_rows),
        },
        metadata,
        settings,
        state,
        notes=[
            "Sign-test hidden-bias bounds do not estimate an effect or certify removal of confounding.",
            "Exact zero differences are reported and excluded from the informative sign count.",
        ],
    )


def _draws(differences, method, draws, seed, max_work, initial_work):
    n = len(differences)
    count = 2**n if method == "exact" else draws
    m.work(
        initial_work + count * max(n, 1) * 4,
        max_work,
        "complete sample/topology and paired sign-flip enumeration/simulation",
    )
    plan = plan_workspace(
        "paired sign-flip draws and saved assignment state",
        {
            "assignment_and_statistic_buffers": 20 * count * max(n, 1),
            "all_draws_and_packed_saved_assignments": 16 * count + 4 * count * n,
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if method == "exact":
        bits = (
            (
                torch.arange(count, dtype=torch.int64, device="cpu")[:, None]
                >> torch.arange(n, dtype=torch.int64, device="cpu")[None, :]
            )
            & 1
        ).bool()
    else:
        bits = torch.randint(
            0, 2, (count, n), generator=generator, dtype=torch.int64, device="cpu"
        ).bool()
    # The randomization statistic is a within-pair difference averaged over
    # all pairs. A sharp additive null imputes the opposite assignment by
    # negating each observed contrast adjusted by the declared null effect.
    statistics = ((bits.to(torch.float64) * 2 - 1) * differences).mean(dim=1)
    assignments = ["".join("1" if value else "0" for value in row) for row in bits.tolist()]
    return statistics, assignments, plan.record()


@m.procedure
def paired_randomization(
    data: Any,
    y: str,
    treatment: str,
    pair: str,
    *,
    design: str,
    null_effect: float = 0.0,
    alternative: str = "two-sided",
    method: str = "exact",
    draws: int = 9999,
    seed: int = 1729,
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Fisher paired-randomization test for a constant additive sharp null.

    ``design='paired_randomized'`` explicitly declares independent, equiprobable
    treatment randomization within each supplied pair. Pair labels alone do not
    establish that design. Exact inference enumerates all 2**n_pairs assignments.
    Monte Carlo draws independent assignments with a private Torch generator;
    p=(extreme+1)/(draws+1), with the observed assignment included by the +1.
    All assignments, statistics, counts, settings and sample positions persist.
    ``draws`` and ``seed`` control Monte Carlo only; exact inference uses no RNG.
    Nonzero contrasts whose contributions to the average statistic underflow
    are refused; rescale the outcome and declared null effect together.
    """
    m.options(device, weights, max_work)
    c.check_choice(design, "design", ("paired_randomized",))
    c.check_choice(alternative, "alternative", ("greater", "less", "two-sided"))
    c.check_choice(method, "method", ("exact", "monte_carlo"))
    null_effect = c.check_number(null_effect, "null_effect")
    draws = c.check_count(draws, "draws", maximum=1_000_000)
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    differences, pair_rows, metadata = _sample(data, y, treatment, pair, missing, max_work)
    adjusted = differences - null_effect
    if not bool(torch.isfinite(adjusted).all()):
        raise AnalysisError(
            "numerical_failure", "Null-adjusted pair contrasts exceed finite float64 range."
        )
    # A nonzero contrast lost by division would turn genuinely distinct
    # assignments into numerical ties and could change an exact p-value.
    # This guard concerns float64 underflow, rather than balanced contrasts:
    # their observed mean may be exactly zero while every contribution remains
    # representable. No positive-magnitude epsilon removes ordinary small data.
    if bool(((adjusted != 0) & (adjusted / len(adjusted) == 0)).any()):
        raise AnalysisError(
            "numerical_failure",
            "A nonzero paired contrast underflows on the average-statistic scale; rescale the outcome and null effect together.",
        )
    observed = float(adjusted.mean())
    if not math.isfinite(observed):
        raise AnalysisError(
            "numerical_failure", "The observed paired statistic exceeds finite float64 range."
        )
    initial_work = metadata["n_input"] * 5
    statistics, assignments, plan = _draws(adjusted, method, draws, seed, max_work, initial_work)
    if not bool(torch.isfinite(statistics).all()):
        raise AnalysisError(
            "numerical_failure", "A randomization statistic exceeds finite float64 range."
        )
    tolerance = 32 * torch.finfo(torch.float64).eps * float(adjusted.abs().mean())
    if alternative == "greater":
        extreme = statistics >= observed - tolerance
    elif alternative == "less":
        extreme = statistics <= observed + tolerance
    else:
        extreme = statistics.abs() >= abs(observed) - tolerance
    total, count = len(statistics), int(extreme.sum())
    p_value = count / total if method == "exact" else (count + 1) / (total + 1)
    # Conditional binomial simulation uncertainty of the plus-one estimate.
    # This is a Monte Carlo SE, never an estimated causal-effect SE.
    mc_se = None if method == "exact" else math.sqrt(p_value * (1 - p_value) * total) / (total + 1)
    settings = dict(
        y=y,
        treatment=treatment,
        pair=pair,
        design=design,
        null_effect=null_effect,
        alternative=alternative,
        method=method,
        draws=draws,
        seed=seed,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    state = dict(
        target="paired Fisher constant-additive sharp-null randomization p-value",
        pair_rows=pair_rows,
        differences=differences.tolist(),
        null_adjusted_differences=adjusted.tolist(),
        observed_statistic=observed,
        randomization_statistics=statistics.tolist(),
        assignment_bits=assignments,
        assignment_encoding="one character per persisted pair in pair_rows; 1 retains observed assignment, 0 swaps assignment",
        extreme=extreme.tolist(),
        n_extreme=count,
        n_assignments=total,
        p_value=p_value,
        monte_carlo_standard_error=mc_se,
        monte_carlo_maximum_standard_error=None
        if method == "exact"
        else math.sqrt(total / 4) / (total + 1),
        monte_carlo_uncertainty="plug-in binomial MCSE plus worst-case binomial MCSE bound; neither is a treatment-effect standard error"
        if method == "monte_carlo"
        else None,
        statistic_tie_tolerance=tolerance,
        simulation_resource_plan=plan,
        planned_work=initial_work + total * len(adjusted) * 4,
        covariance=None,
        standard_error=None,
        df=None,
        confidence_interval=None,
        inference="exact finite assignment enumeration"
        if method == "exact"
        else "private-RNG Monte Carlo with observed-assignment plus-one correction",
        assumptions=[
            "independently randomized equiprobable treatment within each pair",
            "constant additive treatment effect null for every unit",
            "no interference across units",
            "whole-pair outcome availability is fixed independently of treatment assignment",
        ],
        zero_difference_pairs=int((adjusted == 0).sum()),
    )
    draws_rows = [
        [i, value, bool(is_extreme)]
        for i, (value, is_extreme) in enumerate(zip(statistics.tolist(), extreme.tolist()))
    ]
    return m.result(
        "paired_randomization",
        {
            "test": m.frame(
                [[observed, p_value, count, total, mc_se]],
                columns=[
                    "statistic",
                    "p_value",
                    "n_extreme",
                    "n_assignments",
                    "monte_carlo_standard_error",
                ],
            ),
            "pairs": _pair_table(pair_rows),
            "draws": m.frame(draws_rows, columns=["draw", "statistic", "extreme"]),
        },
        metadata,
        settings,
        state,
        notes=[
            "Assignment-based inference tests a constant additive sharp null under the declared paired randomized design.",
            "Exact enumeration uses no simulation; Monte Carlo reports sampling uncertainty separately from causal inference.",
        ],
    )
