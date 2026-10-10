"""Locally seeded independent joint draws from a complete finite BMA posterior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd
import torch

from . import bma as b, bma_common as c, bma_kernels as k, bma_query as q

SCHEMA = "openecon.bayesian_bma_draws.v1"
ARRAY_KEYS = ("model_uniform", "model_mask", "beta", "sigma_squared", "mean_draws", "outcome_draws")


@c.cpu_call
def arrays(parent, draws, seed, source):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = k.tensor(parent["results"]["model_weights"])
    cumulative = torch.cumsum(weights, 0)
    cumulative[-1] = 1.0
    uniform = torch.rand((draws,), dtype=k.DT, device="cpu", generator=generator)
    choices = torch.searchsorted(cumulative, uniform, right=True)
    total = len(parent["components"][0]["location"])
    beta = torch.zeros((draws, total), dtype=k.DT, device="cpu")
    sigma = torch.empty(draws, dtype=k.DT, device="cpu")
    for i, v in enumerate(parent["components"]):
        rows = torch.nonzero(choices == i).flatten()
        count = len(rows)
        if not count:
            continue
        posterior = v["posterior"]
        gamma = torch._standard_gamma(
            torch.full((count,), posterior["shape"], dtype=k.DT, device="cpu"), generator=generator
        )
        variance = posterior["scale"] / gamma
        normals = torch.randn(
            (count, len(v["indices"])), dtype=k.DT, device="cpu", generator=generator
        )
        coefficient = k.tensor(posterior["mean"]) + variance.sqrt()[:, None] * (
            normals @ torch.linalg.cholesky(k.tensor(posterior["conditional_scale_matrix"])).T
        )
        beta[
            rows[:, None], torch.tensor(v["indices"], dtype=torch.int64, device="cpu")[None, :]
        ] = coefficient
        sigma[rows] = variance
    means = outcomes = None
    blocks = [beta, sigma]
    if source is not None:
        _, design = c.source_matrices(source, outcome=False)
        means = beta @ design.T
        noise = torch.randn(means.shape, dtype=k.DT, device="cpu", generator=generator)
        outcomes = means + sigma.sqrt()[:, None] * noise
        blocks.extend((means, outcomes))
    if bool((sigma <= 0).any()) or not all(bool(torch.isfinite(v).all()) for v in blocks):
        c.fail(
            "A seeded posterior draw exceeds supported float64 range; no draw is discarded.",
            "numerical_failure",
        )
    return dict(
        model_uniform=uniform.tolist(),
        model_mask=choices.tolist(),
        beta=beta.tolist(),
        sigma_squared=sigma.tolist(),
        mean_draws=None if means is None else means.tolist(),
        outcome_draws=None if outcomes is None else outcomes.tolist(),
    )


def draw_admission(result, draws, total, queries):
    c.keys(result, ARRAY_KEYS, "complete BMA draw arrays")
    c.array(result["model_uniform"], (draws,), "recorded model uniforms")
    c.array(result["model_mask"], (draws,), "recorded model choices", integers=True)
    c.array(result["beta"], (draws, total), "joint embedded coefficient draws")
    c.array(result["sigma_squared"], (draws,), "joint variance draws")
    for key in ("mean_draws", "outcome_draws"):
        if queries:
            c.array(result[key], (draws, queries), key)
        elif result[key] is not None:
            c.fail("Unqueried draw state cannot contain query arrays.")


@c.cpu_call
def replay(saved):
    raw = c.load(saved, SCHEMA)
    c.keys(
        raw,
        (
            "schema",
            "posterior",
            "posterior_digest",
            "draws",
            "seed",
            "query_source",
            "results",
            "digest",
        ),
        "BMA draw state",
    )
    draws = c.integer(raw["draws"], "saved draws", 1, 10000)
    seed = c.integer(raw["seed"], "saved seed")
    parent, spec, n = q.header(raw["posterior"])
    source = raw["query_source"]
    queries = 0
    if source is not None:
        if not isinstance(source, Mapping):
            c.fail("Saved draw query source must be a mapping.")
        queries = c.integer(source.get("n"), "draw query rows", 1, c.MAX_ROWS)
        positions = source.get("positions")
        if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= queries:
            c.fail("Saved draw query positions are invalid.")
        queries = len(positions)
    c.plan(n, spec, queries=queries, draws=draws)
    draw_admission(raw["results"], draws, 1 + len(spec["forced"]) + len(spec["optional"]), queries)
    if source is not None:
        q.query_admission(source, parent, spec, n)
    parent = b.replay(parent)
    if raw["posterior_digest"] != parent["digest"]:
        c.fail("BMA draw target lineage disagrees.")
    expected = arrays(parent, draws, seed, source)
    c.same(raw["results"], expected, "complete locally seeded mixture draws")
    if raw["digest"] != c.digest({key: value for key, value in raw.items() if key != "digest"}):
        c.fail("Complete BMA draw digest disagrees.")
    return raw


class BMADraws(c.TypedState):
    """Independent model/coefficient/variance draws with one shared model per joint query draw."""

    schema_version: Literal["openecon.bayesian_bma_draws.v1"] = SCHEMA

    @classmethod
    def _replay(cls, value):
        return replay(value)

    @c.cpu_call
    def summary(self):
        """Return exact seed-order model masks, embedded coefficients and variance draws."""
        raw = replay(self)
        spec = raw["posterior"]["spec"]
        r = raw["results"]
        frame = pd.DataFrame(
            r["beta"],
            columns=["Intercept", *spec["forced"], *spec["optional"]],
            index=pd.RangeIndex(raw["draws"], name="draw"),
        )
        frame["model_mask"] = r["model_mask"]
        frame["sigma_squared"] = r["sigma_squared"]
        frame.attrs.update(
            seed=raw["seed"],
            draw_kind="independent finite proper-NIG model mixture",
            posterior_digest=raw["posterior_digest"],
            mean_draws=r["mean_draws"],
            outcome_draws=r["outcome_draws"],
            query_source=raw["query_source"],
        )
        return frame


@c.cpu_call
def bayes_bma_draws(saved, *, draws=1000, seed=0, data=None, missing="raise") -> BMADraws:
    """Draw model masks then joint coefficients/variance and optional indexed outcomes using one local seed.

    All query rows in a draw share the sampled model and coefficients. These are
    independent analytic draws: no chains, warmup or convergence claims exist.
    """
    draws = c.integer(draws, "draws", 1, 10000)
    seed = c.integer(seed, "seed")
    if missing not in ("raise", "drop"):
        c.fail("Draw query missing must be raise or drop.", "invalid_option")
    parent, spec, n = q.header(saved)
    source = None
    queries = 0
    if data is not None:
        queries = c.resident(data, [*spec["forced"], *spec["optional"]])
        c.plan(
            n,
            spec,
            queries=queries,
            draws=draws,
            index_bytes=c.index_buffer_bytes(data.index, budget_bytes=spec["max_bytes"]),
        )
        source = c.capture(
            data, [*spec["forced"], *spec["optional"]], missing, {**spec, "source_n": n}, query=True
        )
        q.query_admission(source, parent, spec, n)
    else:
        c.plan(n, spec, draws=draws)
    parent = b.replay(parent)
    result = arrays(parent, draws, seed, source)
    body = dict(
        schema=SCHEMA,
        posterior=parent,
        posterior_digest=parent["digest"],
        draws=draws,
        seed=seed,
        query_source=source,
        results=result,
    )
    body["digest"] = c.digest(body)
    return BMADraws(payload=body)


def bayes_bma_draws_restore(saved) -> BMADraws:
    """Restore complete model choices and seeded joint draws with no optimizer, search or replacement draws."""
    return BMADraws(payload=replay(saved))
