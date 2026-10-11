"""CR2 study-cluster covariance with Satterthwaite and HTZ small-sample tests.

Bias-reduced linearization (Bell and McCaffrey 2002) for the fitted
inverse-variance working model, following Tipton (2015, Psychological
Methods 20:375-393) and Pustejovsky and Tipton (2018, Journal of Business &
Economic Statistics 36:672-683; doi:10.1080/07350015.2016.1247004). With
working covariance ``Theta_j = V_j`` (the fitted marginal block) and
``W_j = V_j^-1``, the adjustment is ``A_j = L_j (L_j'(Theta_j - X_j M X_j')
L_j)^-1/2 L_j'`` where ``Theta_j = L_j L_j'`` and ``M = (X'WX)^-1``. Single
coefficients/contrasts use the Satterthwaite degrees of freedom and joint
hypotheses the Hotelling-T² approximation with estimated degrees of freedom
(HTZ). Conventions match clubSandwich ``vcovCR(type = "CR2")``,
``coef_test(test = "Satterthwaite")`` and ``Wald_test(test = "HTZ")`` for
rma.mv fits without user weights: https://github.com/cran/clubSandwich.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import f_sf, t_isf, t_sf
from openecon.resources import plan_workspace

from . import multilevel_kernels
from .common import level_check
from .dependent import SCHEMA as DEPENDENT_SCHEMA
from .dependent import load_state as load_dependent
from .dependent_post import _contrasts, _null
from .multilevel import SCHEMA as MULTILEVEL_SCHEMA
from .multilevel import load_state as load_multilevel

SOURCE = "https://github.com/cran/clubSandwich"
REFERENCES = ["Pustejovsky and Tipton (2018), doi:10.1080/07350015.2016.1247004",
              "Tipton and Pustejovsky (2015), doi:10.3102/1076998615606099",
              "Tipton (2015), doi:10.1037/met0000011"]


def _working_model(result):
    """Return padded working blocks, design, residuals, labels and model covariance."""
    raw = result.attrs.get("prediction_state") if isinstance(result, TableSet) else result
    schema = raw.get("schema") if isinstance(raw, dict) else None
    if schema == MULTILEVEL_SCHEMA:
        state, values = load_multilevel(result)
        packing = values["packing"]
        weight = packing.weight
        theta = values["blocks"]-torch.diag_embed(1-weight)
        theta = theta+state["sigma2"][0]*(weight[:, :, None]*weight[:, None, :])+state["sigma2"][1]*torch.diag_embed(weight)
        theta = theta+torch.diag_embed(1-weight)
        x, y, beta, bread = values["x"], values["y"], values["beta"], values["covariance"]
        model = f"three-level ({state['method']})"
    elif schema == DEPENDENT_SCHEMA:
        state, values = load_dependent(result)
        index = values["cluster_index"]
        packing = multilevel_kernels.Packing(index, state["n_studies"])
        s = values["s"]
        if state["model"] == "effect":
            s = s+state["tau2"]*torch.eye(len(index), dtype=torch.float64)
        elif state["model"] == "study":
            s = s+state["tau2"]*(index[:, None] == index[None, :]).to(torch.float64)
        theta = packing.matrix(s)
        x, y, beta, bread = values["x"], values["y"], values["beta"], values["fit_covariance"]
        model = f"dependent {state['model']} ({state['method']})"
    else:
        raise AnalysisError("invalid_result_state", "meta_cr2 requires a saved meta_multilevel or meta_dependent fit with complete state.")
    return state, packing, theta, x, y, beta, bread, model


def _inverse_root(matrix, what):
    eigenvalues, vectors = torch.linalg.eigh((matrix+matrix.transpose(-1, -2))/2)
    largest = eigenvalues.abs().amax(-1, keepdim=True).clamp_min(1e-300)
    if bool((eigenvalues <= largest*1e-10).any()):
        raise AnalysisError("degenerate_inference", f"The {what} is singular; CR2 is undefined without a generalized inverse, which is not substituted.")
    return (vectors*eigenvalues.rsqrt()[..., None, :])@vectors.transpose(-1, -2)


def _satterthwaite(d, ell):
    """Scalar Satterthwaite df from per-cluster d_j and l_j (Pustejovsky-Tipton 2018 eq. 11)."""
    gram = torch.einsum("ja,jb->ab", ell, ell)
    norms = (ell*ell).sum(1)
    expectation = float((d-norms).sum())
    square = float((d*d).sum()-2*(d*norms).sum()+(gram*gram).sum())
    return expectation, expectation**2/square


def _htz(d, ell, wald, q):
    """Hotelling-T² Z (HTZ) test with Wishart-matched degrees of freedom."""
    omega = (d-torch.einsum("jsa,jta->jst", ell, ell)).sum(0)
    root = _inverse_root(omega, "expected CR2 contrast covariance")
    dt = root@d@root
    lt = torch.einsum("st,jta->jsa", root, ell)
    gram = torch.einsum("jsa,jtb->stab", lt, lt)
    inner = torch.einsum("jsa,jta->jst", lt, lt)
    t1 = (torch.einsum("jst,jts->st", dt, dt)-torch.einsum("jst,jts->st", dt, inner)
          - torch.einsum("jst,jts->st", inner, dt)+torch.einsum("stab,tsab->st", gram, gram))
    diag_d = dt.diagonal(dim1=-2, dim2=-1)
    diag_i = inner.diagonal(dim1=-2, dim2=-1)
    t2 = (torch.einsum("js,jt->st", diag_d, diag_d)-torch.einsum("js,jt->st", diag_d, diag_i)
          - torch.einsum("js,jt->st", diag_i, diag_d)+torch.einsum("stab,stab->st", gram, gram))
    nu = q*(q+1)/float((t1+t2).sum())
    df2 = nu-q+1
    delta = max(df2/nu, 0.0)
    statistic = delta*wald/q
    p_value = f_sf(statistic, q, df2) if df2 > 0 and math.isfinite(statistic) else None
    return {"wald_statistic": wald, "F_statistic": statistic, "multiplier": delta, "df1": q, "df2": df2, "nu": nu,
            "p_value": p_value, "status": "ok" if p_value is not None else "undefined: nu-q+1 <= 0"}


@resident_cpu
@torch.no_grad()
def meta_cr2(result, *, contrasts=None, null=None, level=None):
    """CR2 study-robust coefficients with Satterthwaite t and HTZ joint tests.

    Accepts a saved meta_multilevel or meta_dependent fit; the fitted marginal
    study blocks are the inverse-variance working model. Coefficients and
    variance components are not refitted. Reports CR2 covariance, per-term
    Satterthwaite df, the HTZ test of all moderators and optional labelled
    contrasts. Singular adjustment blocks are rejected (no generalized inverse).
    """
    state, packing, theta, x, y, beta, bread, model = _working_model(result)
    level = level_check(state["level"] if level is None else level)
    groups, size, p = packing.groups, packing.size, x.shape[1]
    if groups < 2:
        raise AnalysisError("insufficient_clusters", "CR2 inference requires at least two studies.")
    resource = plan_workspace("meta CR2 small-sample inference", {
        "padded_working_blocks_factors_and_adjustments": 8*10*groups*size*size,
        "cluster_projection_products": 8*8*groups*size*p,
        "cluster_contrast_tensors": 8*6*groups*p*p+8*4*p**4,
    }).record()
    xp, residual = packing.vector(x), packing.vector(y-x@beta)
    factor, info = torch.linalg.cholesky_ex(theta)
    if bool((info != 0).any()):
        raise AnalysisError("numerical_failure", "A working study covariance block is not positive definite.")
    inverse = torch.cholesky_inverse(factor)
    projected = theta-torch.einsum("gmp,pq,gnq->gmn", xp, bread, xp)
    core = factor.transpose(-1, -2)@projected@factor
    adjustment = factor@_inverse_root(core, "CR2 adjustment block (a study has full leverage)")@factor.transpose(-1, -2)
    estimating = torch.einsum("gmp,gmn,gnk->gpk", xp, inverse, adjustment)
    projection = torch.einsum("pq,gqm->gpm", bread, estimating)
    scores = torch.einsum("gpm,gm->gp", projection, residual)
    covariance = scores.T@scores
    covariance = (covariance+covariance.T)/2
    if not bool(torch.isfinite(covariance).all()) or float(torch.linalg.eigvalsh(covariance)[0]) <= float(covariance.diagonal().max())*1e-12:
        raise AnalysisError("degenerate_inference", "Study contributions do not identify a full-rank CR2 covariance; no ridge or pseudoinverse is used.")
    gm = torch.einsum("gpm,gmn,gqn->gpq", projection, theta, projection)
    root_bread = torch.linalg.cholesky(bread)
    ell = torch.einsum("gpm,gmq,qa->gpa", projection, xp, root_bread)
    terms = state["terms"]
    se = covariance.diagonal().sqrt()
    rows = []
    for k, term in enumerate(terms):
        expectation, df = _satterthwaite(gm[:, k, k], ell[:, k, :])
        if not math.isfinite(df) or df <= 0:
            raise AnalysisError("degenerate_inference", "A Satterthwaite degrees-of-freedom estimate is not positive.")
        statistic = float(beta[k]/se[k])
        c = t_isf((1-level)/2, df)
        rows.append({"term": term, "estimate": float(beta[k]), "std_error": float(se[k]), "statistic": statistic,
                     "df": df, "p_value": 2*t_sf(abs(statistic), df), "ci_low": float(beta[k]-c*se[k]),
                     "ci_high": float(beta[k]+c*se[k]), "expected_variance": expectation})
    tests = []
    indices = list(range(int(state["intercept"]), p))
    if indices:
        selector = torch.eye(p, dtype=torch.float64)[indices]
        tests.append({"test": "joint moderators", **_joint(selector, torch.zeros(len(indices), dtype=torch.float64),
                                                           beta, covariance, gm, ell)})
    tables = {"coefficients": table(rows), "covariance": table(covariance.tolist(), columns=terms, index=terms)}
    if contrasts is not None:
        labels, matrix = _contrasts(contrasts, terms)
        nulls = _null(null, labels)
        tests.append({"test": "contrasts", **_joint(matrix, nulls, beta, covariance, gm, ell)})
        records = []
        for label, weights, value in zip(labels, matrix, nulls.tolist()):
            estimate = float(weights@beta)
            standard = float(weights@covariance@weights)**.5
            expectation, df = _satterthwaite(torch.einsum("p,gpq,q->g", weights, gm, weights), torch.einsum("p,gpa->ga", weights, ell))
            statistic = (estimate-value)/standard
            c = t_isf((1-level)/2, df)
            records.append({"contrast": label, "estimate": estimate, "null": value, "std_error": standard,
                            "statistic": statistic, "df": df, "p_value": 2*t_sf(abs(statistic), df),
                            "ci_low": estimate-c*standard, "ci_high": estimate+c*standard})
        tables["contrasts"] = table(records)
        tables["contrast_weights"] = table(matrix.tolist(), columns=terms, index=labels)
    if tests:
        frame = table(tests)
        tables["tests"] = frame.astype(object).where(frame.notna(), None)
    tables["clusters"] = table({"study": state["studies"], "n_effects": packing.counts.tolist(),
                                "adjustment_min_eigenvalue": torch.linalg.eigvalsh(core).amin(-1).tolist()})
    return TableSet(tables, title="CR2 study-robust inference with Satterthwaite/HTZ tests", procedure="meta_cr2",
                    base_procedure=state.get("schema"), working_model=model, level=level, n_effects=state["n_effects"],
                    n_studies=groups, base_sample_hash=state["sample_hash"], base_state_checksum=state["checksum"],
                    correction="CR2", target="fitted marginal covariance (inverse-variance working model)",
                    refitted=False, source=SOURCE, references=REFERENCES, resource_plan=resource,
                    notes=["Coefficients and variance components are the saved fit; only uncertainty is replaced.",
                           "Satterthwaite and HTZ degrees of freedom are approximations evaluated under the working model.",
                           "Adjustment blocks with full leverage are rejected rather than pseudo-inverted."])


def _joint(matrix, nulls, beta, covariance, gm, ell):
    q = matrix.shape[0]
    difference = matrix@beta-nulls
    contrast = matrix@covariance@matrix.T
    eigenvalues = torch.linalg.eigvalsh((contrast+contrast.T)/2)
    if float(eigenvalues[0]) <= float(eigenvalues[-1])*1e-12:
        raise AnalysisError("singular_covariance", "The CR2 contrast covariance is singular.")
    wald = float(difference@torch.linalg.solve(contrast, difference))
    d = torch.einsum("sp,gpq,tq->gst", matrix, gm, matrix)
    projected = torch.einsum("sp,gpa->gsa", matrix, ell)
    return {"q": q, **_htz(d, projected, wald, q)}
