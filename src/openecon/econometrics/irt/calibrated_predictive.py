"""Exact prediction and draws for a declared finite, fixed-calibration IRT prior."""

from __future__ import annotations

import copy
import json
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary, summary_state
from . import calibrated


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise AnalysisError("invalid_options", f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def _device(device):
    if type(device) is not str or device != "cpu":
        raise AnalysisError("unsupported_device", "Calibrated IRT prediction supports device='cpu' only.")


def _selected(state, items):
    names = state["bank"]["items"]
    if items is None:
        items = names.copy()
    if (not isinstance(items, (list, tuple)) or not 1 <= len(items) <= len(names)
            or any(not isinstance(v, str) for v in items) or len(set(items)) != len(items)
            or any(v not in names for v in items)):
        raise AnalysisError("invalid_items", "Supply distinct known item names in the requested prediction order.")
    return list(items), [names.index(v) for v in items]


def _sample(state):
    return table([dict(position=i, source_index=index,
                       observed_items=sum(v >= 0 for v in state["responses"][i]),
                       missing_items=sum(v < 0 for v in state["responses"][i]))
                  for i, index in enumerate(state["indices"])],
                 columns=["position", "source_index", "observed_items", "missing_items"])


def _portable_output(state, layout, *, item_text_per_person=0):
    """Conservative escaped full-JSON bound before numerical reconstruction.

    Each numeric cell/index needs <=32 JSON characters; primitive source labels
    and item names are counted using their actual JSON escapes. 256 KiB covers
    the bounded attrs/column names/settings outside the complete input state.
    No complete table is truncated to meet an export limit.
    """
    n = len(state["responses"])
    q, j = len(state["support"]), len(state["bank"]["items"])
    label_bytes = sum(len(json.dumps(v, ensure_ascii=True, allow_nan=False)) for v in state["indices"])
    bank_names = sum(len(json.dumps(v, ensure_ascii=True)) for v in state["bank"]["items"])
    # Closed primitive state shapes bound encoding without first creating a
    # potentially large full-state JSON string under a small caller budget.
    state_bytes = 16384 + 32 * (n * (q + j + 2) + 3*q + 96) + label_bytes + 2*bank_names
    estimate = 256 * 1024 + state_bytes
    estimate += sum(n * rows * (cells + 1) * 32 for rows, cells in layout)
    estimate += label_bytes * sum(rows for rows, _ in layout) + n * item_text_per_person
    if estimate > 32 * 1024**2:
        raise AnalysisError("resource_limit", "Complete escaped predictive summary may exceed 32 MiB; reduce people, draws, selected items or label lengths.")
    return estimate


def _result(tables, state, method, settings, title, notes):
    notes = [
        "All item calibration parameters are supplied and fixed; their estimation uncertainty is excluded.",
        "The caller-declared finite prior is the actual latent distribution; no continuous-prior quadrature equivalence is claimed.",
        "Partial and all-missing response patterns retain their original positions and source indices.",
    ] + notes
    output = TableSet(tables, title=title, schema="openecon.irt.calibrated.predictive.v1",
                      method=method, posterior_state=copy.deepcopy(state), settings=settings,
                      device="cpu", dtype="float64", stata_parity_validated=False, notes=notes)
    # All settings/notes persist in generic summary JSON and ordinary table
    # cells; frame-only attrs are intentionally outside that export contract.
    saved_summary(output)
    for name, frame in list(output.items()):
        output[name] = table(frame.to_numpy().tolist(), columns=list(frame.columns), index=list(frame.index))
    output = TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)
    summary_state(output)  # Every successful result must actually be portable.
    return output


def _admit(posterior, *, max_bytes, max_work, extra_work, extra_buffers):
    # Primitive-only decoding precedes all downstream dimensions and allocation.
    # _posterior admits BOTH validation and downstream work/buffers before any tensor.
    state = calibrated._posterior_input(posterior, max_bytes=max_bytes)
    n, q = len(state["responses"]), len(state["support"])
    return calibrated._posterior(posterior, max_bytes=max_bytes, max_work=max_work,
                                  extra_work=extra_work(state, n, q),
                                  extra_buffers=extra_buffers(state, n, q))


@resident_cpu
def irt_plausible_values(posterior, *, draws=19, seed=83, max_work=300_000_000,
                         max_bytes=128 * 1024**2, device="cpu"):
    """Draw from each exact finite IRT posterior using a private CPU generator.

    Supplied fixed bank and declared finite prior; preserves every partial or
    all-missing person. 1..99 draws, 0..2**63-1 seed, <=1000 people. Returns every
    support index and latent value, source sample and settings; generic full
    summary JSON/LaTeX persistence. These conditional draws do not establish
    population-model plausible-value inference. CPU float64; no global RNG use.
    """
    _device(device)
    draws = _integer(draws, "draws", 1, 99)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    state = _admit(posterior, max_bytes=max_bytes, max_work=max_work,
                   extra_work=lambda s, n, q: 8 * n * draws * q,
                   extra_buffers=lambda s, n, q: {
                       "private posterior draw uniforms indices and values": n * draws * 8 * 6,
                       "posterior draw cumulative probabilities": n * q * 8 * 3,
                       "complete predictive JSON serialization": 2 * _portable_output(s, [(draws, 5), (1, 4)]),
                   })
    w = torch.tensor(state["posterior"], dtype=torch.float64, device="cpu")
    cumulative = w.cumsum(1)
    cumulative[:, -1] = 1.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand((len(w), draws), generator=generator, dtype=torch.float64, device="cpu")
    indices = torch.searchsorted(cumulative.contiguous(), u.contiguous(), right=True)
    support = torch.tensor(state["support"], dtype=torch.float64, device="cpu")
    values = support[indices]
    rows = [dict(position=i, source_index=state["indices"][i], draw=d + 1,
                 support_index=int(indices[i, d]), theta=float(values[i, d]))
            for i in range(len(w)) for d in range(draws)]
    settings = dict(draws=draws, seed=seed, distribution="exact declared finite posterior",
                    sampling="private CPU uniform inverse CDF", max_work=max_work, max_bytes=max_bytes)
    return _result({"draws": table(rows), "sample": _sample(state),
                    "settings": table([settings])}, state, "plausible_values", settings,
                   "Conditional fixed-bank finite IRT posterior draws", [
                       "Each draw independently samples a declared support point by posterior inverse CDF.",
                       "No population/background model or population-level multiple-imputation inference is supplied.",
                   ])


def _prediction_state(posterior, items, *, max_bytes, max_work, pmf=False, max_support=256):
    primitive = calibrated._posterior_input(posterior, max_bytes=max_bytes)
    names, selected = _selected(primitive, items)
    n, q = len(primitive["responses"]), len(primitive["support"])
    if n > 128:
        raise AnalysisError("resource_limit", "Predictive tables support at most 128 people.")
    definitions = [primitive["bank"]["definitions"][j] for j in selected]
    j, k = len(selected), sum(v["K"] for v in definitions)
    width = 1 + sum(max(v["scores"]) for v in definitions)
    if pmf and width > max_support:
        raise AnalysisError("resource_limit", "Full raw score support exceeds max_support; no support is truncated.")
    extra_work = 20 * (q * k + n * q * (k + j * j))
    buffers = {
        "replicate item probabilities and conditional score moments": 8 * q * (k + j * 5) * 4,
        "complete posterior replicate score covariance and means": 8 * n * (j * j + j) * 8,
        "centered conditional replicate score moments": 8 * n * q * j * 3,
        "complete marginal category probabilities": 8 * n * k * 4,
    }
    if pmf:
        extra_work += 20 * (q * width * k + n * q * width)
        buffers["conditional convolution and posterior total-score PMFs"] = 8 * width * (q * 5 + n * 8)
        output_bytes = _portable_output(primitive, [(width, 6), (1, 8), (1, 4)])
    else:
        lengths = [len(json.dumps(name, ensure_ascii=True)) for name in names]
        item_text = sum(d["K"] * length for d, length in zip(definitions, lengths))
        item_text += (1 + 2 * j) * sum(lengths)
        output_bytes = _portable_output(primitive, [(k, 6), (j, 5), (j*j, 5), (1, 4)],
                                        item_text_per_person=item_text)
    buffers["complete predictive JSON serialization"] = 2 * output_bytes
    state = calibrated._posterior(posterior, max_bytes=max_bytes, max_work=max_work,
                                  extra_work=extra_work, extra_buffers=buffers)
    return state, names, selected, width


def _moments(state, selected):
    theta = torch.tensor(state["support"], dtype=torch.float64, device="cpu")
    w = torch.tensor(state["posterior"], dtype=torch.float64, device="cpu")
    all_logp = calibrated._logp(state["bank"], theta)
    probabilities = [all_logp[j].exp() for j in selected]
    means, variances, constants = [], [], []
    for j, p in zip(selected, probabilities):
        scores = torch.tensor(state["bank"]["definitions"][j]["scores"], dtype=torch.float64, device="cpu")
        constant = bool((scores == scores[0]).all())
        mu = torch.full((len(p),), float(scores[0]), dtype=torch.float64, device="cpu") if constant else p @ scores
        means.append(mu)
        variances.append((p * (scores[None, :] - mu[:, None]).square()).sum(1))
        constants.append(constant)
    conditional_means = torch.stack(means, 1)
    conditional_variances = torch.stack(variances, 1)
    mean = w @ conditional_means
    for a, constant in enumerate(constants):
        if constant:
            mean[:, a] = conditional_means[0, a]
    centered = conditional_means[None, :, :] - mean[:, None, :]
    covariance = torch.einsum("nq,nqj,nqk->njk", w, centered, centered)
    diagonal = torch.arange(len(selected), device="cpu")
    covariance[:, diagonal, diagonal] += w @ conditional_variances
    covariance = (covariance + covariance.transpose(1, 2)) / 2
    return w, probabilities, mean, covariance


@resident_cpu
def irt_predictive(posterior, *, items=None, max_work=300_000_000,
                   max_bytes=128 * 1024**2, device="cpu"):
    """Exact joint score moments for new replicated fixed-bank IRT responses.

    Mix item category probabilities over each declared finite posterior. Return
    all category probabilities, expected caller-defined item scores and full
    score covariance including latent induced cross-item dependence. Predicts
    future replicates conditional on observed answers; not uncertainty in those
    observed answers. Optional distinct known item subset preserves its order.
    <=128 people, <=16 items; CPU float64, early combined work/workspace limits.
    """
    _device(device)
    state, names, selected, _ = _prediction_state(posterior, items, max_bytes=max_bytes, max_work=max_work)
    w, probabilities, mean, covariance = _moments(state, selected)
    marginals = [w @ p for p in probabilities]
    category_rows, mean_rows, covariance_rows = [], [], []
    for i, index in enumerate(state["indices"]):
        for a, (name, j, p) in enumerate(zip(names, selected, marginals)):
            for category, score in enumerate(state["bank"]["definitions"][j]["scores"]):
                category_rows.append(dict(position=i, source_index=index, item=name,
                                          category=category, score=score, probability=float(p[i, category])))
            mean_rows.append(dict(position=i, source_index=index, item=name,
                                  expected_score=float(mean[i, a]), variance=float(covariance[i, a, a])))
            for b, name2 in enumerate(names):
                covariance_rows.append(dict(position=i, source_index=index, item1=name, item2=name2,
                                            covariance=float(covariance[i, a, b])))
    settings = dict(items=names, prediction="new conditional replicate", covariance="complete item-score covariance",
                    max_work=max_work, max_bytes=max_bytes)
    return _result({"probabilities": table(category_rows), "means": table(mean_rows),
                    "covariance": table(covariance_rows), "sample": _sample(state)},
                   state, "replicate_prediction", settings, "Finite IRT posterior replicate prediction", [
                       "Conditional item responses are independent given theta; posterior mixing induces cross-item score dependence.",
                       "Covariance is allowed to be positive semidefinite and singular; no positive-definite repair is applied.",
                       "Nominal category scores are an explicit caller scoring rule and do not imply category ordering.",
                   ])


@resident_cpu
def irt_test_score_distribution(posterior, *, items=None, level=0.95, max_support=256,
                                max_work=300_000_000, max_bytes=128 * 1024**2, device="cpu"):
    """Exact full posterior predictive PMF of a future replicate total score.

    Convolve caller-defined nonnegative integer category scores at each theta,
    then mix using posterior masses; preserves latent induced dependence.
    Duplicate scores aggregate and gapped support keeps zero-probability cells.
    Returns full PMF/CDF, tails, moments and inverse-CDF equal-tail quantiles.
    These predictive intervals are not ability confidence intervals. <=128 people,
    <=16 items, complete support admitted before tensors; CPU float64.
    """
    _device(device)
    max_support = _integer(max_support, "max_support", 1, 256)
    try:
        valid_level = not isinstance(level, bool) and isinstance(level, Real) and math.isfinite(level) and 0 < level < 1
    except (OverflowError, TypeError, ValueError):
        valid_level = False
    if not valid_level:
        raise AnalysisError("invalid_options", "level must be finite and strictly between zero and one.")
    level = float(level)
    alpha = (1 - level) / 2
    if not 0 < alpha < .5 < 1 - alpha < 1:
        raise AnalysisError("invalid_options", "level must yield distinct interior equal-tail probabilities in float64.")
    state, names, selected, width = _prediction_state(posterior, items, max_bytes=max_bytes,
                                                    max_work=max_work, pmf=True, max_support=max_support)
    w, probabilities, mean_items, covariance = _moments(state, selected)
    conditional = torch.ones((len(state["support"]), 1), dtype=torch.float64, device="cpu")
    for j, p in zip(selected, probabilities):
        scores = state["bank"]["definitions"][j]["scores"]
        convolved = torch.zeros((len(w[0]), conditional.shape[1] + max(scores)), dtype=torch.float64, device="cpu")
        for category, score in enumerate(scores):
            convolved[:, score:score + conditional.shape[1]] += conditional * p[:, category:category + 1]
        conditional = convolved
    pmf = w @ conditional
    if not bool(torch.isfinite(pmf).all()) or not torch.allclose(pmf.sum(1), torch.ones(len(w), dtype=torch.float64), atol=2e-12, rtol=0):
        raise AnalysisError("numerical_failure", "Posterior replicate PMF did not retain finite unit mass.")
    cdf = pmf.cumsum(1)
    cdf[:, -1] = 1.0
    tails = pmf.flip(1).cumsum(1).flip(1)
    scores = torch.arange(width, dtype=torch.float64, device="cpu")
    mean = pmf @ scores
    variance = (pmf * (scores[None, :] - mean[:, None]).square()).sum(1)
    if (not torch.allclose(mean, mean_items.sum(1), atol=2e-10, rtol=2e-12)
            or not torch.allclose(variance, covariance.sum((1, 2)), atol=2e-9, rtol=2e-12)):
        raise AnalysisError("numerical_failure", "Full score PMF and joint replicate moments disagree.")
    rows, summary = [], []
    for i, index in enumerate(state["indices"]):
        low = int(torch.searchsorted(cdf[i], torch.tensor(alpha, dtype=torch.float64, device="cpu")))
        high = int(torch.searchsorted(cdf[i], torch.tensor(1 - alpha, dtype=torch.float64, device="cpu")))
        for score in range(width):
            rows.append(dict(position=i, source_index=index, score=score, probability=float(pmf[i, score]),
                             cdf=float(cdf[i, score]), tail_ge=float(tails[i, score])))
        summary.append(dict(position=i, source_index=index, expected_score=float(mean[i]),
                            variance=float(variance[i]), sd=float(variance[i].sqrt()),
                            quantile_low=low, quantile_high=high, level=level))
    settings = dict(items=names, level=level, max_support=max_support,
                    mixture="conditional convolution at each theta then posterior mixing", max_work=max_work, max_bytes=max_bytes)
    return _result({"distribution": table(rows), "summary": table(summary), "sample": _sample(state)},
                   state, "test_score_distribution", settings, "Finite IRT posterior replicate total-score distribution", [
                       "Complete raw support from zero through the maximum caller-defined total score is retained, including impossible cells.",
                       "The PMF uses conditional convolution before posterior mixing; convolving marginal item probabilities is incorrect.",
                       "Quantiles use the smallest integer score whose CDF reaches each equal-tail probability.",
                       "Intervals describe future replicate score uncertainty conditional on the fixed bank and responses.",
                   ])
