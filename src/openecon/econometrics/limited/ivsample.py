"""Sample, first stage and Newey's two-step estimator for ``ivprobit`` / ``ivtobit``.

Sample
------
``w = [x1, y2]`` are the regressors of the outcome equation (included
exogenous variables with the constant first, then the endogenous regressors)
and ``z = [x1, x2]`` all exogenous variables (``x2`` = the excluded
instruments). Collinear exogenous regressors and instruments are omitted left
to right; an endogenous regressor that is an exact linear combination of the
exogenous variables, or fewer instruments than endogenous regressors, is an
error. With a constant, the exogenous variables and the endogenous regressors
are centred at their (weighted) means for all computations;
``EndogenousSample.reported`` maps the constants of the outcome equation and
of the reduced forms back exactly (``a = a_c - m_x'g - m_y'b`` and
``pi_0 = pi_0c + m_y - m_z'pi``), so the reported model is the one specified.

First stage
-----------
Least squares (QR) of every endogenous regressor on ``z``. For each one
``extra['first_stage']`` reports R-squared, adjusted R-squared, the partial
R-squared and the F test of the excluded instruments
``F = [(SSR_r - SSR) / m] / [SSR / (N - kz)]`` with ``m`` instruments, and the
root mean squared error.

Newey's minimum chi-squared estimator (``method='twostep'``)
-----------------------------------------------------------
Stata's ``ivprobit, twostep`` / ``ivtobit, twostep`` (Newey 1987), with
``V-hat`` the first-stage residuals and ``Pi-hat`` the first-stage
coefficients:

1. Reduced-form probit (tobit) of ``y1`` on ``[z, V-hat]``: coefficients
   ``alpha`` of ``z`` with covariance block ``J_aa^-1`` and ``lambda`` of
   ``V-hat``.
2. Two-stage conditional maximum likelihood (Rivers-Vuong 2SIV): probit
   (tobit) of ``y1`` on ``[w, V-hat]`` gives a consistent ``beta`` for the
   endogenous regressors; the Wald test that its ``V-hat`` coefficients are
   zero is the test of exogeneity.
3. Least squares of ``y2 (lambda - beta)`` on ``z``; its classical covariance
   ``s^2 (z'z)^-1``, ``s^2 = SSR / (N - kz)``, is added:
   ``Omega = J_aa^-1 + s^2 (z'z)^-1``.
4. With ``D = [I_1, Pi-hat]`` (``z D = [x1, z Pi-hat]``):

       d = (D' Omega^-1 D)^-1 D' Omega^-1 alpha,     Var(d) = (D' Omega^-1 D)^-1.

The coefficients of ``ivprobit`` are those of the index conditional on the
first-stage errors (normalized by the conditional standard deviation), so
they differ from the maximum likelihood coefficients by a scale factor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, kernel_call, wald_test
from openecon.econometrics.limited.common import Weights, centre, likelihood_weights
from openecon.econometrics.registry import role_columns
from openecon.engines.distributions import f_sf
from openecon.engines.linalg import cholesky_inverse, collinear_columns, least_squares
from openecon.models import ModelSpec


class EndogenousSample:
    """Estimation sample and design blocks of a model with endogenous regressors."""

    def __init__(self, spec: ModelSpec, data: Any, command: str):
        self.command = command
        endogenous = role_columns(spec, "endogenous")
        instruments = role_columns(spec, "instruments")
        seen = {spec.outcome: "the outcome"}
        for group, names in (("an exogenous regressor", spec.predictors),
                             ("an endogenous regressor", endogenous),
                             ("an instrument", instruments)):
            for name in names:
                if name in seen:
                    raise AnalysisError(
                        "invalid_spec", f"{command}: '{name}' is both {seen[name]} and {group}. "
                        "Exogenous regressors are instruments for themselves and must not be "
                        "repeated; an endogenous regressor cannot be its own instrument.")
                seen[name] = group
        if set(endogenous) & set(spec.categorical):
            raise AnalysisError("invalid_spec", f"{command}: endogenous regressors must be "
                                "continuous columns, not categorical ones.")
        frame = ModelFrame(spec, data)
        self.frame: ModelFrame = frame
        self.weights: Weights = likelihood_weights(frame)
        w = self.weights.user
        screen = self.weights.for_screen()
        exog = frame.drop_collinear(frame.design(), screen)
        excluded = frame.design(instruments, intercept=False)
        combined = Design(torch.cat([exog.x, excluded.x], dim=1), [*exog.terms, *excluded.terms],
                          {**exog.categories, **excluded.categories}, exog.intercept)
        if len(combined.terms) != len(set(combined.terms)):
            raise AnalysisError("duplicate_terms", "Instrument names collide with regressor "
                                "terms. Rename the affected columns.")
        combined = frame.drop_collinear(combined, screen)
        k1 = len(exog.terms)
        if combined.terms[:k1] != exog.terms:
            raise AnalysisError("singular_design", f"{command}: the exogenous regressors are "
                                "collinear once the instruments are added; remove the redundant "
                                "columns.")
        # Centre the exogenous variables and the endogenous regressors (see module notes).
        y2 = frame.matrix(endogenous)
        self.z_centres = centre(combined, screen)
        self.y2_centres = torch.zeros(len(endogenous), dtype=torch.float64)
        if combined.intercept:
            self.y2_centres = (w[:, None] * y2).sum(dim=0) / w.sum()
            y2 = y2 - self.y2_centres
        self.exog, self.z = combined.select(range(k1)), combined
        self.endogenous = endogenous
        self.instruments = combined.terms[k1:]
        self.y2 = y2
        self.k1, self.kz, self.p = k1, len(combined.terms), len(endogenous)
        if len(self.instruments) < self.p:
            raise AnalysisError(
                "underidentified",
                f"{command}: {self.p} endogenous regressor(s) need at least as many excluded "
                f"instruments, but only {len(self.instruments)} remain after removing collinear "
                "ones. Add instruments that are not regressors of the outcome equation.")
        _, dependent = kernel_call(collinear_columns, torch.cat([combined.x, self.y2], dim=1), w)
        if any(i < self.kz for i in dependent):
            raise AnalysisError("singular_design", f"{command}: the exogenous variables are "
                                "numerically collinear; rescale or remove the redundant columns.")
        if dependent:
            names = ", ".join(endogenous[i - self.kz] for i in dependent)
            raise AnalysisError(
                "collinear_endogenous",
                f"{command}: {names} is an exact linear combination of the exogenous variables "
                "(or of the other endogenous regressors), so it has no first-stage error. Treat "
                "it as exogenous or remove it.")
        self.terms = [*exog.terms, *endogenous]
        self.categories = combined.categories
        self._first: Any = None

    def reported(self, params: Tensor, covariance: Tensor, *, reduced_forms: bool
                 ) -> tuple[Tensor, Tensor]:
        """Undo the centring of the data: estimates and covariance for the columns as given.

        ``params`` starts with the outcome equation ``(a_c, g, b)`` and, when
        ``reduced_forms`` is true, continues with one block ``(pi_0c, pi)`` per
        endogenous regressor; further (ancillary) parameters are unchanged.
        The map is affine, ``J params + c``, so the covariance is ``J V J'``.
        """
        if not self.exog.intercept:
            return params, covariance
        k1, kz, p = self.k1, self.kz, self.p
        kw = k1 + p
        jacobian = torch.eye(len(params), dtype=torch.float64)
        shift = torch.zeros(len(params), dtype=torch.float64)
        jacobian[0, 1:k1] = -self.z_centres[1:k1]
        jacobian[0, k1:kw] = -self.y2_centres
        if reduced_forms:
            for j in range(p):
                start = kw + j * kz
                jacobian[start, start + 1:start + kz] = -self.z_centres[1:]
                shift[start] = self.y2_centres[j]
        return jacobian @ params + shift, jacobian @ covariance @ jacobian.T

    @property
    def w(self) -> Tensor:
        """Regressors of the outcome equation ``[x1, y2]``."""
        return torch.cat([self.exog.x, self.y2], dim=1)

    def first_stage(self):
        """Least squares of the endogenous regressors on all exogenous variables (cached)."""
        if self._first is None:
            self._first = kernel_call(least_squares, self.z.x, self.y2, self.weights.user,
                                      drop_collinear=False)
        return self._first

    def first_stage_summary(self) -> dict[str, Any]:
        """Per endogenous regressor: fit and the F test of the excluded instruments."""
        first, w, nobs = self.first_stage(), self.weights.user, self.weights.nobs
        total = float(w.sum())
        if self.k1:
            restricted = kernel_call(least_squares, self.exog.x, self.y2, w,
                                     drop_collinear=False).ssr
        else:
            restricted = (w[:, None] * self.y2.square()).sum(dim=0)
        if self.exog.intercept:
            centred = self.y2 - (w[:, None] * self.y2).sum(dim=0) / total
            spread = (w[:, None] * centred.square()).sum(dim=0)
        else:
            spread = (w[:, None] * self.y2.square()).sum(dim=0)
        m, df2 = self.kz - self.k1, nobs - self.kz
        summary = {}
        for j, name in enumerate(self.endogenous):
            ssr, ssr_r, tss = float(first.ssr[j]), float(restricted[j]), float(spread[j])
            statistic = ((ssr_r - ssr) / m) / (ssr / df2) if df2 > 0 and ssr > 0 else None
            r2 = 1 - ssr / tss if tss > 0 else None
            summary[name] = {
                "r_squared": r2,
                "adjusted_r_squared": None if r2 is None or df2 <= 0 else
                1 - (1 - r2) * (nobs - int(self.exog.intercept)) / df2,
                "partial_r_squared": 1 - ssr / ssr_r if ssr_r > 0 else None,
                "f_statistic": statistic, "f_df1": m, "f_df2": df2,
                "f_p_value": None if statistic is None else f_sf(statistic, m, df2),
                "rmse": math.sqrt(ssr / df2) if df2 > 0 else None,
            }
        return summary

    def record(self) -> dict[str, Any]:
        """Model description stored in ``extra``."""
        return {"exogenous": self.exog.terms, "endogenous": self.endogenous,
                "instruments": self.instruments, "n_endogenous": self.p,
                "n_instruments": len(self.instruments), "first_stage": self.first_stage_summary()}


# ``fit(design) -> (coefficients incl. ancillary, covariance, log likelihood)`` of the
# outcome model (probit or tobit) on an arbitrary design.
OutcomeFit = Callable[[Tensor, str], tuple[Tensor, Tensor, float]]


def newey_two_step(sample: EndogenousSample, outcome_fit: OutcomeFit) -> dict[str, Any]:
    """Newey's (1987) minimum chi-squared estimator (see the module notes)."""
    command = sample.command
    weights, nobs = sample.weights.user, sample.weights.nobs
    z, kz, k1, p = sample.z.x, sample.kz, sample.k1, sample.p
    kw = k1 + p
    if nobs <= kz + p + 1:
        raise AnalysisError("insufficient_observations", f"{command}: the two-step estimator "
                            f"needs more observations ({nobs}) than first-stage parameters.")
    first = sample.first_stage()
    resid = first.resid
    reduced, reduced_cov, _ = outcome_fit(torch.cat([z, resid], dim=1),
                                          "reduced-form model of the two-step estimator")
    alpha, lam = reduced[:kz], reduced[kz:kz + p]
    conditional, conditional_cov, log_likelihood = outcome_fit(
        torch.cat([sample.w, resid], dim=1), "two-stage conditional (2SIV) model")
    beta = conditional[k1:kw]
    auxiliary = kernel_call(least_squares, z, sample.y2 @ (lam - beta), weights,
                            drop_collinear=False)
    omega = reduced_cov[:kz, :kz] + auxiliary.xtx_inv * (float(auxiliary.ssr) / (nobs - kz))
    distance = torch.zeros((kz, kw), dtype=torch.float64)
    distance[:k1, :k1] = torch.eye(k1, dtype=torch.float64)
    distance[:, k1:] = first.beta
    try:
        omega_inverse = kernel_call(cholesky_inverse, omega)
        covariance = kernel_call(cholesky_inverse, distance.T @ omega_inverse @ distance)
    except AnalysisError as exc:
        raise AnalysisError(
            "singular_covariance",
            f"{command}: the minimum chi-squared weighting matrix is singular; the instruments "
            "do not identify the coefficients of the endogenous regressors (rank condition). "
            "Check that the excluded instruments are relevant.") from exc
    delta = covariance @ (distance.T @ (omega_inverse @ alpha))
    exogeneity = wald_test(conditional, conditional_cov, range(kw, kw + p),
                           label="Wald test of exogeneity (first-stage residual coefficients = 0)")
    return {"delta": delta, "covariance": covariance, "exogeneity": exogeneity,
            "alpha": alpha, "lambda": lam, "conditional": conditional[:kw + p],
            "conditional_log_likelihood": log_likelihood}


def structural_index(sample: EndogenousSample, delta: Tensor) -> Tensor:
    """``w'd`` for the (centred) regressors and the matching, not yet uncentred, ``d``."""
    k1 = sample.k1
    return sample.exog.x @ delta[:k1] + sample.y2 @ delta[k1:k1 + sample.p]
