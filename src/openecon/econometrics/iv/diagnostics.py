"""First-stage, weak-identification, overidentification and endogeneity diagnostics.

All statistics are built from the two projections of ``kernels.project`` plus
least squares on blocks of q or L2 columns; nothing here refactors the n-by-K
design. N is the number of observations as the estimator counts it, K1/q/L2
the exogenous, endogenous and excluded-instrument counts, K = K1 + q,
L = K1 + L2 and ``a`` the degrees of freedom of absorbed effects (zero for
ivregress).

First stage (Stata's ``estat firststage``), per endogenous regressor x:
    R2 = 1 - RSS_Z/TSS, adjusted with (N - c - a)/(N - L - a);
    partial R2 = 1 - RSS_Z/RSS_1 (RSS after Z = [X1 Z2] and after X1 only);
    F(L2, N-L-a) = {(RSS_1 - RSS_Z)/L2} / {RSS_Z/(N-L-a)}, or with a robust,
    cluster or HAC model covariance the Wald F of the excluded instruments
    under that covariance with regress's factors (second df G - 1 if clustered);
    Shea's partial R2 = [(X'WX)^-1]_xx / [(Xhat'W Xhat)^-1]_xx.

Weak identification:
    Cragg-Donald Wald F = (N-L-a)/L2 * min eig{(X2'M_Z X2)^-1 X2'(P_Z - P_1)X2}
    (Stata's minimum eigenvalue statistic);
    Kleibergen-Paap rk Wald F (robust/cluster/HAC) = rk/N * (N-L-a)/L2, with
    clusters rk/(N-1) * (N-L-a) * (G-1)/G / L2 (ivreg2's scaling), where rk is
    the rank statistic of ``kernels.rank_test_rows`` under the model covariance.

Overidentification, df = L2 - q:
    2SLS, nonrobust: Sargan = N u'P_Z u / u'u and Basmann = Sargan (N-L-a)/(N-Sargan);
    2SLS, robust: Wooldridge's score test s'M^-1 s with s = sum_i w_i k_i u_i, k
    the residuals of the excluded instruments on Xhat and M the robust/cluster/
    HAC covariance of those moments (N - RSS of 1 on u k in the White case);
    LIML: Anderson-Rubin N ln(kappa) and Basmann F = (kappa - 1)(N-L-a)/(L2-q);
    GMM: Hansen's J = g(b)'S^-1 g(b).

Endogeneity (H0: the endogenous regressors are exogenous), q restrictions, with
v the first-stage residuals and the augmented regression y on [X, v]:
    Durbin = N (RSS_ols - RSS_aug)/RSS_ols, chi2(q);
    Wu-Hausman = {(RSS_ols - RSS_aug)/q} / {RSS_aug/(N-K-q-a)}, F(q, N-K-q-a);
    robust score (Wooldridge): s'M^-1 s for the moments r_i u_ols,i with
    r = M_X v, chi2(q); robust regression F: Wald F of the coefficients of v in
    the augmented regression under the robust covariance with regress's factors;
    GMM: C = J_e - J_c, the J of the model treating the regressors as
    exogenous (instruments [Z X2]) minus the J of the estimated model, both with
    the moment covariance of the former, chi2(q).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, kernel_call, wald_test
from openecon.econometrics.iv import kernels
from openecon.econometrics.iv.common import (
    Blocks, Estimate, Setting, covariance, gmm_system, moment_covariance, moment_meat,
    sum_of_squares,
)
from openecon.engines.distributions import chi2_sf, f_sf
from openecon.engines.linalg import (
    cholesky_inverse, least_squares, wald_statistic, weighted_crossprod,
)

# First-stage residuals below this fraction of the regressor's own variation mean the
# instruments reproduce the regressor: tests built on those residuals are undefined.
_NO_RESIDUAL = 1e-20


def chi2_entry(statistic: float, df: int, label: str, **extra: Any) -> dict[str, Any]:
    statistic = max(0.0, statistic)
    return {"statistic": statistic, "df": df, "p_value": kernel_call(chi2_sf, statistic, df),
            "distribution": "chi2", "label": label, **extra}


def f_entry(statistic: float, df: int, df2: float, label: str, **extra: Any) -> dict[str, Any]:
    statistic = max(0.0, statistic)
    return {"statistic": statistic, "df": df, "df2": df2,
            "p_value": kernel_call(f_sf, statistic, df, df2), "distribution": "F",
            "label": label, **extra}


def unavailable(label: str, note: str, distribution: str = "chi2") -> dict[str, Any]:
    return {"statistic": None, "df": 0, "p_value": None, "distribution": distribution,
            "label": label, "note": note}


def _score_test(frame: ModelFrame, setting: Setting, rows: Tensor) -> tuple[float, int]:
    """``s'M^- s`` for the moment rows: s their weighted sum, M their covariance."""
    w = setting.weights
    total = rows.sum(dim=0) if w is None else (rows * w[:, None]).sum(dim=0)
    kind = "robust" if setting.kind == "nonrobust" else setting.kind
    return kernel_call(wald_statistic, total, moment_meat(frame, setting, rows, kind=kind))


class Diagnostics:
    """Diagnostics of one IV fit; ``tests`` and ``first_stage`` are filled by ``run``."""

    def __init__(self, frame: ModelFrame, setting: Setting, y: Tensor, blocks: Blocks,
                 est: Estimate):
        self.frame, self.setting, self.y, self.blocks, self.est = frame, setting, y, blocks, est
        self.proj, self.w = est.proj, setting.weights
        self.n, self.a = setting.nobs, setting.absorbed
        self.k1, self.q, self.l2 = self.proj.k1, self.proj.q, self.proj.l2
        self.k, self.l = self.k1 + self.q, self.k1 + self.l2
        self.df_first = self.n - self.l - self.a
        self.robust = setting.kind != "nonrobust"
        self.tests: dict[str, Any] = {}
        self.first_stage: list[dict[str, Any]] = []

    def run(self) -> Diagnostics:
        if self.df_first < self.q:
            # The q-by-q first-stage residual moments are singular: nothing can be tested.
            self.frame.warn("First-stage, weak-identification, overidentification and "
                            "endogeneity diagnostics were not computed: the first stage has "
                            f"only {self.df_first} residual degree(s) of freedom.")
            return self
        # OLS of [y, v] on X after partialling out X1: u_ols and r = M_X v.
        self.ols = kernel_call(kernels.solve, self.proj.x2p,
                               torch.cat([self.proj.y_p[:, None], self.proj.e], dim=1), self.w,
                               "endogenous regressors")
        noise = weighted_crossprod(self.proj.e, self.w).diagonal()
        signal = weighted_crossprod(self.proj.x2p, self.w).diagonal()
        self.exact_first_stage = bool((noise <= _NO_RESIDUAL * signal).any())
        self._first_stage()
        if self.est.method in {"fuller", "kclass"}:
            self.tests["overid_unavailable"] = unavailable(
                "Test of overidentifying restrictions",
                "No calibrated overidentification diagnostic is assigned to this k-class fit; structural weak-IV inference is available through iv_weak_test.")
        else:
            self._overidentification()
        if self.est.method not in {"liml", "fuller", "kclass"}:
            self._endogeneity()
        self._weak_identification()
        return self

    # ---- first stage ---------------------------------------------------------------

    def _first_stage(self) -> None:
        proj, w, l2 = self.proj, self.w, self.l2
        constant = int(self.blocks.exog.intercept)
        robust = None
        if self.robust:
            robust = kernel_call(kernels.solve, proj.z2p, proj.x2p, w, "excluded instruments")
        for j, term in enumerate(self.blocks.endog.terms):
            rss, rss_1 = float(proj.first.ssr[j]), float(proj.partial.ssr[j])
            tss = sum_of_squares(self.blocks.x2[:, j], w, centered=bool(constant))
            r2 = 1 - rss / tss if tss > 0 else None
            row: dict[str, Any] = {
                "term": term, "r_squared": r2,
                "adjusted_r_squared": None if r2 is None else
                1 - (1 - r2) * (self.n - constant - self.a) / self.df_first,
                "partial_r_squared": 1 - rss / rss_1 if rss_1 > 0 else None,
                "shea_partial_r_squared":
                    float(self.ols.xtx_inv[j, j] / self.est.tsls.inner[j, j]),
            }
            if robust is None:
                statistic = ((rss_1 - rss) / l2) / (rss / self.df_first) if rss > 0 else None
                row.update({"f_statistic": statistic, "df": l2, "df2": self.df_first,
                            "p_value": None if statistic is None else
                            kernel_call(f_sf, max(statistic, 0.0), l2, self.df_first),
                            "covariance": "nonrobust"})
            else:
                v, info = covariance(self.frame, self.setting, score_x=proj.z2p,
                                     resid=proj.e[:, j], bread=robust.xtx_inv, k=self.l,
                                     regress=True)
                test = wald_test(robust.beta[:, j], v, range(l2),
                                 df_resid=info["df_inference"])
                row.update({"f_statistic": test["statistic"], "df": test["df"],
                            "df2": test.get("df2"), "p_value": test["p_value"],
                            "covariance": self.setting.kind})
            self.first_stage.append(row)

    # ---- weak identification -------------------------------------------------------

    def _weak_identification(self) -> None:
        label = "Cragg-Donald Wald F (minimum eigenvalue statistic)"
        note = "weak-identification statistic: compare with Stock-Yogo critical values"
        if self.exact_first_stage:
            self.tests["cragg_donald"] = unavailable(
                label, "an endogenous regressor is fitted exactly by the instruments", "F")
            return
        try:
            eigenvalue = kernel_call(kernels.cragg_donald, self.proj, self.est.tsls.g, self.w)
        except AnalysisError as exc:
            self.tests["cragg_donald"] = unavailable(label, str(exc), "F")
            return
        self.tests["cragg_donald"] = {
            "statistic": eigenvalue * self.df_first / self.l2, "df": self.l2,
            "df2": self.df_first, "p_value": None, "distribution": "F", "label": label,
            "note": note}
        if not self.robust:
            return
        label = "Kleibergen-Paap rk Wald F"
        try:
            smallest, rows = kernel_call(kernels.rank_test_rows, self.proj, self.w)
            omega_inv = kernel_call(cholesky_inverse,
                                    moment_meat(self.frame, self.setting, rows))
        except AnalysisError:
            self.tests["kleibergen_paap_rk_f"] = unavailable(
                label, "the covariance of the rank statistic is singular", "F")
            return
        rk = smallest * smallest * float(omega_inv[0, 0])
        if self.setting.kind == "cluster":
            groups = min(count for _, count in self.setting.clusters)
            statistic = rk / (self.n - 1) * self.df_first * (groups - 1) / groups / self.l2
        else:
            statistic = rk / self.n * self.df_first / self.l2
        self.tests["kleibergen_paap_rk_f"] = {
            "statistic": statistic, "df": self.l2, "df2": self.df_first, "p_value": None,
            "distribution": "F", "label": label, "rk_wald_chi2": rk,
            "rk_df": self.l2 - self.q + 1, "note": note}

    # ---- overidentification --------------------------------------------------------

    def _overidentification(self) -> None:
        est, df, n = self.est, self.l2 - self.q, self.n
        names = {"2sls": "overid_score" if self.robust else "overid_sargan",
                 "liml": "anderson_rubin", "gmm": "hansen_j"}
        if df == 0:
            self.tests[names[est.method]] = unavailable(
                "Test of overidentifying restrictions",
                "exactly identified: there are no overidentifying restrictions")
            return
        if est.method == "gmm":
            kind = est.gmm["wmatrix"]
            label = "Hansen's J test of overidentifying restrictions" if kind != "unadjusted" \
                else "Sargan test of overidentifying restrictions (unadjusted weight matrix)"
            self.tests["hansen_j"] = chi2_entry(est.gmm["j"], df, label, wmatrix=kind)
        elif est.method == "liml":
            self.tests["anderson_rubin"] = chi2_entry(
                n * math.log(est.kappa), df, "Anderson-Rubin LR test of overidentifying "
                "restrictions")
            self.tests["basmann_f"] = f_entry(
                (est.kappa - 1) * self.df_first / df, df, self.df_first,
                "Basmann F test of overidentifying restrictions")
        elif not self.robust:
            z = torch.cat([self.blocks.x1, self.blocks.z2], dim=1)
            moment = weighted_crossprod(z, self.w, est.resid)
            sargan = n * float(moment @ (self.proj.first.xtx_inv @ moment)) / est.rss
            self.tests["overid_sargan"] = chi2_entry(
                sargan, df, "Sargan (score) test of overidentifying restrictions")
            self.tests["overid_basmann"] = chi2_entry(
                sargan * self.df_first / (n - sargan), df,
                "Basmann test of overidentifying restrictions")
        else:
            # Residuals of the excluded instruments on Xhat = [X1, P_Z X2] (rank L2 - q).
            resid = kernel_call(least_squares, est.tsls.g, self.proj.z2p, self.w, tol=0.0).resid
            statistic, rank = _score_test(self.frame, self.setting, resid * est.resid[:, None])
            entry = chi2_entry(statistic, df, "Wooldridge robust score test of overidentifying "
                               "restrictions")
            if rank != df:
                entry["note"] = f"the moment covariance has rank {rank}, not {df}"
            self.tests["overid_score"] = entry

    # ---- endogeneity ---------------------------------------------------------------

    def _endogeneity(self) -> None:
        q, n = self.q, self.n
        gmm = self.est.method == "gmm"
        names = ["endog_c"] if gmm else ["endog_robust_score", "endog_robust_regression"] \
            if self.robust else ["endog_durbin", "endog_wu_hausman"]
        label = "Test that the endogenous regressors are exogenous"
        if self.exact_first_stage:
            for name in names:
                self.tests[name] = unavailable(label, "an endogenous regressor is fitted "
                                               "exactly by the instruments")
            return
        u_ols, r = self.ols.resid[:, 0], self.ols.resid[:, 1:]
        if gmm:
            self._c_statistic(u_ols)
            return
        df2 = self.n - self.k - q - self.a
        try:
            augmented = kernel_call(kernels.solve, r, u_ols, self.w, "first-stage residuals")
        except AnalysisError:
            augmented = None
        if augmented is None or df2 <= 0:
            for name in names:
                self.tests[name] = unavailable(label, "the first-stage residuals are collinear "
                                               "with the regressors or no degrees of freedom "
                                               "remain")
            return
        rss_ols, rss_aug = float(self.ols.ssr[0]), float(augmented.ssr)
        if not self.robust:
            self.tests["endog_durbin"] = chi2_entry(
                n * (rss_ols - rss_aug) / rss_ols, q, "Durbin (score) chi2 test of exogeneity")
            self.tests["endog_wu_hausman"] = f_entry(
                ((rss_ols - rss_aug) / q) / (rss_aug / df2), q, df2,
                "Wu-Hausman F test of exogeneity")
            return
        statistic, rank = _score_test(self.frame, self.setting, r * u_ols[:, None])
        self.tests["endog_robust_score"] = chi2_entry(
            statistic, rank, "Wooldridge robust score chi2 test of exogeneity")
        v, info = covariance(self.frame, self.setting, score_x=r, resid=augmented.resid,
                             bread=augmented.xtx_inv, k=self.k + q, regress=True)
        self.tests["endog_robust_regression"] = wald_test(
            augmented.beta, v, range(q), df_resid=info["df_inference"],
            label="Robust regression-based F test of exogeneity")

    def _c_statistic(self, u_ols: Tensor) -> None:
        """C = J(regressors exogenous) - J(model), both with the former's moment covariance."""
        gmm, w = self.est.gmm, self.w
        x, z, _ = gmm_system(self.blocks, w)
        wide = torch.cat([z, x[:, self.k1:]], dim=1)
        label = "C (difference-in-J) chi2 test of exogeneity"
        try:
            s = moment_covariance(self.frame, self.setting, wide, u_ols,
                                  wmatrix=gmm["wmatrix"], center=gmm["centered"])
            exogenous = kernel_call(kernels.gmm_step, weighted_crossprod(wide, w, x),
                                    weighted_crossprod(wide, w, self.y), s)
            model = kernel_call(kernels.gmm_step, weighted_crossprod(z, w, x),
                                weighted_crossprod(z, w, self.y), s[:self.l, :self.l])
        except AnalysisError:
            self.tests["endog_c"] = unavailable(label, "the moment covariance is singular")
            return
        self.tests["endog_c"] = chi2_entry(exogenous.j - model.j, self.q, label,
                                           wmatrix=gmm["wmatrix"])
