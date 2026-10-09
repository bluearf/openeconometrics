"""Fixed single-target pattern-mixture sensitivity on a declared link scale.

The observed-data model never identifies delta. Numerical kernels are native
CPU float64 Torch; finite logistic/count MH runs make no stationarity claim.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

import pandas as pd
import torch
from pydantic import model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.chained import (
    _finite, _gaussian_draw, _integer, _logit_draw, _normal_parameters, _positive,
)
from openecon.econometrics.mi.common import MAX_COLUMNS, MAX_ROWS, MIResult, _resident, admit, check_seed, make_result
from openecon.resources import plan_workspace

_DTYPE = torch.float64
_MAX_COUNT = 1_000_000
_PLAN_SCOPE = ("estimated live model-input and tensor buffers, retained Python model diagnostics and "
               "serialization copies; excludes caller input, private BLAS workspace and allocator overhead; "
               "not a process-RSS limit")


def _delta(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise AnalysisError("invalid_spec", "delta must be a finite real sensitivity parameter.")
    return float(value)


def _predictors(columns, target, predictors) -> tuple[str, ...]:
    if predictors is None:
        return tuple(name for name in columns if name != target)
    if (isinstance(predictors, (str, bytes)) or not isinstance(predictors, Sequence)
            or any(not isinstance(name, str) for name in predictors)):
        raise AnalysisError("invalid_spec", "predictors must be a sequence of selected column names.")
    names = tuple(predictors)
    if (len(set(names)) != len(names) or target in names
            or any(name not in columns for name in names)):
        raise AnalysisError("invalid_spec", "predictors must contain unique other selected columns.")
    return names


def _preflight(data, columns, target, predictors, *, kind, m, transitions, max_work):
    # Read only resident shape before admit constructs any selected frame/tensor.
    if (not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= MAX_COLUMNS
            or any(not isinstance(name, str) or not name for name in columns)
            or len(set(columns)) != len(columns)):
        raise AnalysisError("invalid_spec", "columns must be 1..16 distinct nonempty names.")
    if not isinstance(target, str) or target not in columns:
        raise AnalysisError("invalid_spec", "target must name one selected column.")
    predictor_names = _predictors(columns, target, predictors)
    n, resident = _resident(data, list(columns))
    if not 1 <= n <= MAX_ROWS:
        raise AnalysisError("mi_shape_limit", "mi_delta requires 1..10000 resident rows; no rows are dropped.")
    p, k = len(columns), len(predictor_names) + 1
    admission_work = n * p**3 + n * p**2 + p**3 + m * n * p
    model_work = n * k**2 + k**3
    draw_work = n * k + n + k**2
    if kind != "normal":
        draw_work += transitions * (n * k + k**2 + n)
    projected_work = admission_work + model_work + m * draw_work
    if projected_work > max_work:
        raise AnalysisError("mi_work_limit", f"mi_delta work {projected_work:,} exceeds max_work={max_work:,}.")
    index_bytes = (6 * int(resident.index.memory_usage(deep=True))
                   if isinstance(resident, pd.DataFrame) else n * 512)
    admitted_bytes = (8 * n * p * 8 + n * p * 4 + 8 * n * 4 + m * n * p * 8
                      + 4 * n * (p + 1) * 8 + 64 * p * p * 8
                      + (m + 2) * n * p * 160 + n * 4096 + index_bytes)
    plan = plan_workspace("fixed single-target pattern-mixture imputation", {
        "admitted panels and immutable result": admitted_bytes,
        "observed and missing designs and factors": 6 * n * k * 8 + 16 * k * k * 8,
        "predictive vectors and local sampler state": 12 * n * 8 + 16 * k * 8,
        # Retain final state only, not all transitions. Count metadata Python
        # objects plus temporary canonical JSON/checksum copies conservatively.
        "saved model draws predictions and checksum copies": m * (8192 + 512 * k + 1024 * n),
    })
    return predictor_names, projected_work, plan


def _saved_number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or (positive and float(value) <= 0)):
        raise ValueError(f"Invalid saved sensitivity {name}.")
    return float(value)


def _saved_integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"Invalid saved sensitivity {name}.")
    return value


def _close(actual, expected, name):
    try:
        if isinstance(expected, torch.Tensor):
            if (not isinstance(actual, (list, tuple))
                    or any(isinstance(value, bool) or not isinstance(value, Real)
                           or not math.isfinite(float(value)) for value in actual)):
                raise ValueError(f"Invalid saved sensitivity {name}.")
            tensor = torch.tensor(actual, dtype=_DTYPE, device="cpu")
            valid = tensor.shape == expected.shape and bool(torch.isfinite(tensor).all())
            if not valid or not torch.allclose(tensor, expected, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Saved sensitivity {name} disagrees with the model.")
        elif (isinstance(actual, bool) or not isinstance(actual, Real)
              or not math.isfinite(float(actual))
              or not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"Saved sensitivity {name} disagrees with the model.")
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid saved sensitivity {name}.") from error


class MIDeltaResult(MIResult):
    """Immutable fixed-delta state with scientific replay validation."""

    @model_validator(mode="after")
    def _sensitivity_integrity(self):
        meta = self.metadata
        n, p, m = len(self.original), len(self.columns), len(self.completed_matrices)
        if self.method != "mi_chained" or meta.get("operation") != "fixed_single_target_pattern_mixture":
            raise ValueError("MIDeltaResult needs a fixed single-target pattern-mixture operation.")
        kind, target = meta.get("kind"), meta.get("target")
        if kind not in ("normal", "logit", "poisson") or target not in self.columns:
            raise ValueError("Invalid saved sensitivity target or kind.")
        delta = _saved_number(meta.get("delta"), "delta")
        if (meta.get("delta_identified_from_observed_data") is not False
                or meta.get("convergence_claim") is not False
                or meta.get("delta_scale") != {"normal": "outcome location", "logit": "log odds", "poisson": "log mean"}[kind]):
            raise ValueError("Saved sensitivity delta or convergence claims are invalid.")
        predictor_map = meta.get("predictors")
        if not isinstance(predictor_map, Mapping) or set(predictor_map) != {target}:
            raise ValueError("Saved sensitivity predictors disagree with the target.")
        predictor_names = predictor_map[target]
        if (not isinstance(predictor_names, tuple) or any(not isinstance(name, str) for name in predictor_names)
                or len(set(predictor_names)) != len(predictor_names)
                or any(name not in self.columns or name == target for name in predictor_names)
                or meta.get("design_terms") != ("intercept", *predictor_names)
                or meta.get("methods") != {target: kind}):
            raise ValueError("Saved sensitivity predictor terms are invalid.")
        k, j = len(predictor_names) + 1, self.columns.index(target)
        mask = tuple(row[j] is None for row in self.original)
        if not any(mask) or all(mask) or any(row[c] is None for row in self.original for c in range(p) if c != j):
            raise ValueError("Saved sensitivity requires exactly one partially observed target.")
        missing_positions = tuple(i for i, absent in enumerate(mask) if absent)
        observed_positions = tuple(i for i, absent in enumerate(mask) if not absent)
        if (meta.get("missing_positions") != missing_positions or meta.get("observed_positions") != observed_positions
                or any(type(i) is not int for i in meta.get("missing_positions", ()))
                or any(type(i) is not int for i in meta.get("observed_positions", ()))):
            raise ValueError("Saved sensitivity row positions disagree with source values.")
        sampler, prior = meta.get("sampler"), meta.get("prior")
        if not isinstance(sampler, Mapping) or not isinstance(prior, Mapping):
            raise ValueError("Saved sensitivity needs its sampler and prior.")
        if (set(sampler) != {"family", "exact_target", "stationarity_claim", "burn", "steps", "proposal_scale"}
                or sampler.get("exact_target") is not True or sampler.get("stationarity_claim") is not False):
            raise ValueError("Saved sensitivity cannot claim finite-chain stationarity.")
        burn = _saved_integer(sampler.get("burn"), "burn", 0, 100_000)
        steps = _saved_integer(sampler.get("steps"), "steps", 0 if kind == "normal" else 1, 100_000)
        if kind == "normal":
            if (burn != 0 or steps != 0 or sampler.get("proposal_scale") is not None
                    or sampler.get("family") != "conjugate Gaussian"
                    or prior != {"family": "p(beta,sigma2) proportional to 1/sigma2", "proper_posterior_conditions":
                                 "full-rank observed design, positive residual SSE and positive residual df"}):
                raise ValueError("Saved normal sensitivity prior or sampler is invalid.")
            prior_scale = None
        else:
            prior_scale = _saved_number(prior.get("scale"), "prior scale", positive=True)
            if not math.isfinite(prior_scale * prior_scale) or prior_scale * prior_scale <= 0:
                raise ValueError("Invalid saved sensitivity prior variance.")
            proposal_scale = _saved_number(sampler.get("proposal_scale"), "proposal scale", positive=True)
            if not math.isfinite(proposal_scale * proposal_scale) or proposal_scale * proposal_scale <= 0:
                raise ValueError("Invalid saved sensitivity proposal variance.")
            if (set(prior) != {"family", "mean", "scale", "includes_intercept", "coefficient_units"}
                    or sampler.get("family") != "symmetric random-walk Metropolis"
                    or prior.get("family") != "independent Gaussian" or type(prior.get("mean")) not in (int, float)
                    or prior.get("mean") != 0 or prior.get("includes_intercept") is not True
                    or prior.get("coefficient_units") != "raw selected predictor units"):
                raise ValueError("Saved discrete sensitivity prior or sampler is invalid.")
        admission_work = n * p**3 + n * p**2 + p**3 + m * n * p
        draw_work = n * k + n + k**2
        if kind != "normal":
            draw_work += (burn + steps) * (n * k + k**2 + n)
        expected_work = admission_work + n * k**2 + k**3 + m * draw_work
        if type(meta.get("projected_work")) is not int or meta.get("projected_work") != expected_work or meta.get("work_estimate") != expected_work:
            raise ValueError("Saved sensitivity work estimate disagrees with the model.")
        expected_buffers = {
            "admitted panels and immutable result": meta["resource_plan"]["estimated_workspace_bytes"],
            "observed and missing designs and factors": 6 * n * k * 8 + 16 * k * k * 8,
            "predictive vectors and local sampler state": 12 * n * 8 + 16 * k * 8,
            "saved model draws predictions and checksum copies": m * (8192 + 512 * k + 1024 * n),
        }
        saved_plan = meta.get("chained_resource_plan", {})
        if (saved_plan.get("operation") != "fixed single-target pattern-mixture imputation"
                or saved_plan.get("buffers") != expected_buffers or saved_plan.get("scope") != _PLAN_SCOPE):
            raise ValueError("Saved sensitivity resource plan disagrees with the model.")
        plan_workspace("fixed single-target sensitivity restoration", expected_buffers)
        # Allocate only after the full dimension/work/workspace replay checks.
        source = torch.tensor([[0.0 if value is None else value for value in row]
                               for row in self.original], dtype=_DTYPE, device="cpu")
        x = torch.cat((torch.ones((n, 1), dtype=_DTYPE, device="cpu"),
                       source[:, tuple(self.columns.index(name) for name in predictor_names)]), dim=1)
        xo, xm = x[list(observed_positions)], x[list(missing_positions)]
        y = source[list(observed_positions), j]
        if kind == "logit" and not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("Saved sensitivity observed binary support is invalid.")
        if kind == "poisson" and not bool(((y >= 0) & (y <= _MAX_COUNT) & (y == y.floor())).all()):
            raise ValueError("Saved sensitivity observed count support is invalid.")
        fit = None
        if kind == "normal":
            try:
                fit = _normal_parameters(xo, y, target)
            except AnalysisError as error:
                raise ValueError("Saved sensitivity Gaussian observed model is invalid.") from error
            observed_model = meta.get("observed_model", {})
            if not isinstance(observed_model, Mapping):
                raise ValueError("Saved sensitivity needs the observed Gaussian model.")
            _close(observed_model.get("beta_hat"), fit.beta_hat, "observed Gaussian coefficients")
            _close(observed_model.get("sse"), float(fit.sse), "observed Gaussian SSE")
            if type(observed_model.get("residual_df")) is not int or observed_model.get("residual_df") != fit.df:
                raise ValueError("Saved sensitivity Gaussian residual df disagrees.")
        diagnostics = meta.get("imputation_diagnostics", ())
        if not isinstance(diagnostics, tuple) or len(diagnostics) != m:
            raise ValueError("Saved sensitivity needs one model draw per imputation.")
        for index, (diagnostic, completed) in enumerate(zip(diagnostics, self.completed_matrices)):
            if (not isinstance(diagnostic, Mapping) or diagnostic.get("imputation") != index + 1
                    or type(diagnostic.get("imputation")) is not int
                    or diagnostic.get("seed") != meta["imputation_seeds"][index]
                    or type(diagnostic.get("seed")) is not int
                    or diagnostic.get("missing_positions") != missing_positions
                    or any(type(i) is not int for i in diagnostic.get("missing_positions", ()))):
                raise ValueError("Saved sensitivity model draw identity disagrees.")
            beta_values = diagnostic.get("coefficient_draw", ())
            if not isinstance(beta_values, tuple) or len(beta_values) != k:
                raise ValueError("Saved sensitivity coefficient dimensions disagree.")
            beta = torch.tensor([_saved_number(value, "coefficient") for value in beta_values],
                                dtype=_DTYPE, device="cpu")
            eta, observed_eta = xm @ beta, xo @ beta
            shifted = eta + delta
            _close(diagnostic.get("base_missing_linear_predictor"), eta, "base missing predictor")
            _close(diagnostic.get("missing_linear_predictor"), shifted, "delta-shifted predictor")
            if kind == "normal":
                sigma2 = _saved_number(diagnostic.get("sigma2_draw"), "Gaussian variance", positive=True)
                if type(diagnostic.get("residual_df")) is not int or diagnostic.get("residual_df") != fit.df:
                    raise ValueError("Saved sensitivity model draw residual df disagrees.")
                _close(diagnostic.get("predictive_mean"), shifted, "Gaussian predictive mean")
                residual = y - observed_eta
                ll = -0.5 * (len(y) * math.log(2 * math.pi * sigma2) + residual.square().sum() / sigma2)
            elif kind == "logit":
                _close(diagnostic.get("predictive_probability"), torch.sigmoid(shifted), "Bernoulli probability")
                if any(completed[position][j] not in (0.0, 1.0) for position in missing_positions):
                    raise ValueError("Saved sensitivity completed binary support is invalid.")
                ll = (y * observed_eta - torch.nn.functional.softplus(observed_eta)).sum()
            else:
                mean = shifted.exp()
                if not bool(torch.isfinite(mean).all()) or bool(((mean <= 0) | (mean > _MAX_COUNT)).any()):
                    raise ValueError("Saved sensitivity predictive mean exceeds the supported domain.")
                _close(diagnostic.get("predictive_mean"), mean, "Poisson predictive mean")
                if any(value < 0 or value > _MAX_COUNT or value != math.floor(value)
                       for value in (completed[position][j] for position in missing_positions)):
                    raise ValueError("Saved sensitivity completed count support is invalid.")
                ll = (y * observed_eta - observed_eta.exp() - torch.lgamma(y + 1)).sum()
            _close(diagnostic.get("observed_log_likelihood"), float(ll), "observed likelihood")
            if kind != "normal":
                mh = diagnostic.get("sampler", {})
                expected_mh = {"burn_proposals", "sampling_proposals", "accepted_burn",
                               "accepted_sampling", "acceptance_rate", "final_log_posterior"}
                if kind == "poisson":
                    expected_mh.add("rejected_nonfinite")
                if not isinstance(mh, Mapping) or set(mh) != expected_mh:
                    raise ValueError("Saved sensitivity MH diagnostics are invalid.")
                if mh.get("burn_proposals") != burn or mh.get("sampling_proposals") != steps:
                    raise ValueError("Saved sensitivity MH proposal counts disagree.")
                for name, upper in (("burn_proposals", burn), ("sampling_proposals", steps),
                                    ("accepted_burn", burn), ("accepted_sampling", steps)):
                    _saved_integer(mh.get(name), name, 0, upper)
                accepted = mh["accepted_burn"] + mh["accepted_sampling"]
                _close(mh.get("acceptance_rate"), accepted / (burn + steps), "MH acceptance rate")
                posterior = ll + (torch.lgamma(y + 1).sum() if kind == "poisson" else 0) - beta.square().sum() / (2 * prior_scale**2)
                _close(mh.get("final_log_posterior"), float(posterior), "MH posterior target")
                if kind == "poisson":
                    _saved_integer(mh.get("rejected_nonfinite"), "rejected_nonfinite", 0, burn + steps - accepted)
        if meta.get("imputation_seeds") != tuple((self.seed + index) % (2**63) for index in range(m)):
            raise ValueError("Saved sensitivity independent chain seeds disagree.")
        contract = meta.get("source_contract", {})
        if (not isinstance(contract, Mapping) or contract.get("delta_adjustment") != "https://amices.org/mice/reference/mice.impute.mnar.html"
                or contract.get("poisson_predictive_model") != "https://mc-stan.org/docs/2_29/stan-users-guide/posterior-prediction-for-regressions.html"
                or contract.get("implementation_parity_claim") is not False):
            raise ValueError("Saved sensitivity source or parity contract is invalid.")
        if meta.get("assumptions") != (
            "fixed single incomplete target with complete selected predictors",
            "caller-declared unidentifiable constant delta",
            "conditional observed working model extends to missing rows with shifted link",
        ):
            raise ValueError("Saved sensitivity conditioning assumptions disagree.")
        return self


def mi_delta(data, columns, *, target, kind, delta, m=5, seed=0, predictors=None,
             prior_scale=2.5, proposal_scale=0.5, mh_burn=100, mh_steps=100,
             max_work=100_000_000) -> MIDeltaResult:
    """Impute one incomplete target with a caller-declared fixed delta.

    ``normal`` shifts the missing conditional mean in outcome units, ``logit``
    shifts log odds, and ``poisson`` shifts log mean. Every other selected
    column must be complete. Delta is not learned from observed data. See
    ``docs/econometrics/mi-sensitivity.md`` for priors, budgets and exclusions.
    """
    seed = check_seed(seed)
    m = _integer(m, "m", 1, 100)
    if not isinstance(kind, str) or kind not in ("normal", "logit", "poisson"):
        raise AnalysisError("invalid_spec", "kind must be 'normal', 'logit', or 'poisson'.")
    delta = _delta(delta)
    prior_scale = _positive(prior_scale, "prior_scale")
    proposal_scale = _positive(proposal_scale, "proposal_scale")
    mh_burn = _integer(mh_burn, "mh_burn", 0, 100_000)
    mh_steps = _integer(mh_steps, "mh_steps", 1, 100_000)
    max_work = _integer(max_work, "max_work", 1, 10_000_000_000)
    predictor_names, projected_work, plan = _preflight(
        data, columns, target, predictors, kind=kind, m=m,
        transitions=mh_burn + mh_steps, max_work=max_work,
    )
    frame, values, missing, admission = admit(data, columns, m=m, max_work=max_work)
    names = tuple(frame.columns)
    j = names.index(target)
    incomplete = tuple(name for column, name in enumerate(names) if bool(missing[:, column].any()))
    if incomplete != (target,):
        raise AnalysisError("invalid_missing_pattern", "Exactly the target must be incomplete; all other selected columns must be complete.")
    mask = missing[:, j]
    n = len(frame)
    n_observed = n - int(mask.sum())
    if n_observed == 0:
        raise AnalysisError("all_missing", "The sensitivity target needs observed outcomes.")
    y = values[~mask, j]
    if kind == "logit" and not bool(((y == 0) | (y == 1)).all()):
        raise AnalysisError("invalid_binary_outcome", "Observed logit target values must be exactly 0 or 1.")
    if kind == "poisson" and not bool(((y >= 0) & (y <= _MAX_COUNT) & (y == y.floor())).all()):
        raise AnalysisError("invalid_count_outcome", "Observed Poisson target values must be integers in [0, 1000000].")
    predictor_indices = tuple(names.index(name) for name in predictor_names)
    design = torch.cat((torch.ones((n, 1), dtype=_DTYPE, device="cpu"),
                        values[:, predictor_indices]), dim=1)
    xo, xm = design[~mask], design[mask]
    fit = _normal_parameters(xo, y, target) if kind == "normal" else None
    missing_positions = torch.nonzero(mask, as_tuple=False)[:, 0].tolist()
    observed_positions = torch.nonzero(~mask, as_tuple=False)[:, 0].tolist()
    chain_seeds = [(seed + imputation) % (2**63) for imputation in range(m)]
    completed, diagnostics = [], []
    try:
        for imputation, chain_seed in enumerate(chain_seeds, 1):
            generator = torch.Generator(device="cpu").manual_seed(chain_seed)
            diagnostic = {"imputation": imputation, "seed": chain_seed,
                          "missing_positions": missing_positions}
            if kind == "normal":
                beta, sigma2 = _gaussian_draw(fit, generator)
                eta = xm @ beta
                prediction = eta + delta
                noise = torch.randn(xm.shape[0], dtype=_DTYPE, device="cpu", generator=generator)
                draw = prediction + sigma2.sqrt() * noise
                residual = y - xo @ beta
                log_likelihood = -0.5 * (n_observed * torch.log(2 * math.pi * sigma2)
                                         + residual.square().sum() / sigma2)
                diagnostic.update({"sigma2_draw": float(sigma2), "residual_df": fit.df,
                                   "predictive_mean": prediction.tolist()})
            else:
                if kind == "logit":
                    beta, sampler = _logit_draw(
                        xo, y, generator, prior_scale=prior_scale, proposal_scale=proposal_scale,
                        burn=mh_burn, steps=mh_steps,
                    )
                else:
                    from openecon.econometrics.mi.discrete import _poisson_draw
                    beta, sampler = _poisson_draw(
                        xo, y, generator, prior_scale=prior_scale, proposal_scale=proposal_scale,
                        burn=mh_burn, steps=mh_steps,
                    )
                eta = xm @ beta
                prediction = eta + delta
                _finite(prediction, "The shifted missing linear predictor")
                observed_eta = xo @ beta
                if kind == "logit":
                    probability = torch.sigmoid(prediction)
                    draw = (torch.rand(probability.shape, dtype=_DTYPE, device="cpu", generator=generator)
                            < probability).to(_DTYPE)
                    log_likelihood = (y * observed_eta - torch.nn.functional.softplus(observed_eta)).sum()
                    diagnostic["predictive_probability"] = probability.tolist()
                else:
                    mean = torch.exp(prediction)
                    _finite(mean, "Poisson missing means")
                    if bool((mean <= 0).any()) or bool((mean > _MAX_COUNT).any()):
                        raise AnalysisError("predictive_mean_limit", "Poisson missing means must be positive and <=1000000; rescale or revise delta.")
                    draw = torch.poisson(mean, generator=generator)
                    if bool((draw > _MAX_COUNT).any()):
                        raise AnalysisError("count_limit", "A Poisson predictive draw exceeds 1000000; no clipping is applied.")
                    log_likelihood = (y * observed_eta - observed_eta.exp() - torch.lgamma(y + 1)).sum()
                    diagnostic["predictive_mean"] = mean.tolist()
                diagnostic["sampler"] = sampler
            _finite(eta, "The base missing linear predictor")
            _finite(prediction, "The shifted missing linear predictor")
            _finite(draw, "Sensitivity imputations")
            _finite(log_likelihood, "The observed-data likelihood")
            diagnostic.update({"coefficient_draw": beta.tolist(),
                               "base_missing_linear_predictor": eta.tolist(),
                               "missing_linear_predictor": prediction.tolist(),
                               "observed_log_likelihood": float(log_likelihood)})
            filled = values.clone()
            filled[mask, j] = draw
            completed.append(filled)
            diagnostics.append(diagnostic)
    except torch.linalg.LinAlgError as exc:
        raise AnalysisError("numerical_failure", "The sensitivity posterior factorization failed.") from exc
    prior = ({"family": "p(beta,sigma2) proportional to 1/sigma2", "proper_posterior_conditions":
              "full-rank observed design, positive residual SSE and positive residual df"}
             if kind == "normal" else {"family": "independent Gaussian", "mean": 0.0,
                                       "scale": prior_scale, "includes_intercept": True,
                                       "coefficient_units": "raw selected predictor units"})
    metadata = {
        **admission, "operation": "fixed_single_target_pattern_mixture", "target": target,
        "kind": kind, "delta": delta, "delta_identified_from_observed_data": False,
        "delta_scale": {"normal": "outcome location", "logit": "log odds", "poisson": "log mean"}[kind],
        "predictors": {target: list(predictor_names)}, "methods": {target: kind},
        "design_terms": ["intercept", *predictor_names], "observed_positions": observed_positions,
        "missing_positions": missing_positions, "prior": prior,
        "sampler": {"family": "conjugate Gaussian" if kind == "normal" else "symmetric random-walk Metropolis",
                    "exact_target": True, "stationarity_claim": False,
                    "burn": mh_burn if kind != "normal" else 0,
                    "steps": mh_steps if kind != "normal" else 0,
                    "proposal_scale": proposal_scale if kind != "normal" else None},
        "convergence_claim": False, "imputation_diagnostics": diagnostics,
        "projected_work": projected_work, "work_estimate": projected_work,
        "chained_resource_plan": {**plan.record(), "scope": _PLAN_SCOPE},
        "assumptions": ["fixed single incomplete target with complete selected predictors",
                        "caller-declared unidentifiable constant delta",
                        "conditional observed working model extends to missing rows with shifted link"],
        "source_contract": {"delta_adjustment": "https://amices.org/mice/reference/mice.impute.mnar.html",
                            "poisson_predictive_model": "https://mc-stan.org/docs/2_29/stan-users-guide/posterior-prediction-for-regressions.html",
                            "implementation_parity_claim": False},
    }
    if fit is not None:
        metadata["observed_model"] = {"beta_hat": fit.beta_hat.tolist(), "sse": float(fit.sse),
                                      "residual_df": fit.df}
    return MIDeltaResult.model_validate(make_result(
        "mi_chained", frame, values, missing, completed, seed, metadata,
        imputation_seeds=chain_seeds,
    ).model_dump())
