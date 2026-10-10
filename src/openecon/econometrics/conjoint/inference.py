"""Complete linear target and iid-respondent mean covariance, without refitting."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines import distributions as dist
from . import common as c
from .model import confidence, inference


@resident_cpu
def conjoint_group_mean(result, *, level=0.95, max_work=100_000_000,
                        device="cpu", weights=None):
    """Mean utilities with full empirical covariance for independent iid respondents.

    The target is the population mean of the fitted respondent coefficients.
    Use sample coefficient covariance / respondent count; within-person fitting
    noise is already part of that dispersion and is not added a second time.
    Marginal t(m-1) intervals are approximate unless respondent estimates are
    Gaussian. This does not infer importance or market-share uncertainty.
    """
    state = c.intact(result, "conjoint_fit")
    level = confidence(level)
    p, m = len(state["terms"]), len(state["subjects"])
    if m < 2:
        raise AnalysisError("insufficient_subjects", "Group sampling covariance needs at least two iid respondents.")
    targets = [(item, value) for item in state["mapping"]
               for value in state["attributes"][item["attribute"]]]
    k = len(targets)
    settings = c.guard("conjoint_group_mean", k, p, m, device=device, weights=weights,
                       max_work=max_work, work=m*p*p+k*p*p+k*k*p,
                       extra_buffers={"complete_utility_covariance": 8*k*k, "utility_map": 8*k*p})
    transform = torch.tensor(state["transform"], dtype=torch.float64)
    beta = torch.stack([transform@torch.tensor(r["parameters_scaled"], dtype=torch.float64)
                        for r in state["subjects"]])
    mean = beta.mean(0)
    centered = beta-mean
    covariance = centered.T@centered/(m*(m-1))
    mapping = torch.stack([c.level_vector(item, value, state["attributes"][item["attribute"]], p)
                           for item, value in targets])
    utilities, utility_covariance = mapping@mean, mapping@covariance@mapping.T
    if not all(torch.isfinite(a).all() for a in (mean, covariance, utilities, utility_covariance)):
        raise AnalysisError("numerical_failure", "Group coefficient/utility moments are nonfinite.")
    columns = ["estimate", "std_error", "df", "t", "p_value", "ci_low", "ci_high"]
    utility_names = [f"utility_{i}" for i in range(k)]
    return c.seal("conjoint_group_mean", {
        "coefficients": table([[term, *inference(float(mean[i]), float(covariance[i, i]), m-1, level)]
                               for i, term in enumerate(state["terms"])], columns=["term", *columns]),
        "covariance": table([[term, *covariance[i].tolist()] for i, term in enumerate(state["terms"])],
                            columns=["term", *state["terms"]]),
        "utilities": table([[utility_names[i], item["attribute"], value,
                             *inference(float(utilities[i]), float(utility_covariance[i, i]), m-1, level)]
                            for i, (item, value) in enumerate(targets)],
                           columns=["target", "attribute", "level", *columns]),
        "utility_covariance": table([[name, *utility_covariance[i].tolist()] for i, name in enumerate(utility_names)],
                                    columns=["target", *utility_names]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"],
        "subjects": [r["subject"] for r in state["subjects"]], "n_subjects": m,
        "terms": state["terms"], "mean_parameters": mean.tolist(), "covariance": covariance.tolist(),
        "level": level, "sampling": "independent iid respondents; empirical fitted-coefficient dispersion / m",
        "settings": settings}, inference="approximate marginal iid-respondent t(m-1); no within-person double counting",
        **settings)


def contrast_spec(contrasts, null, terms):
    if not isinstance(contrasts, dict) or not 1 <= len(contrasts) <= len(terms):
        raise AnalysisError("invalid_contrast", "Declare 1..p named linear contrasts of saved raw coefficients.")
    names, rows, nulls = [], [], []
    if null is not None and (not isinstance(null, dict) or set(null) != set(contrasts)):
        raise AnalysisError("invalid_contrast", "Null must supply exactly the declared contrast names.")
    for name, weights in contrasts.items():
        if not isinstance(name, str) or name in ("subject", "contrast"):
            raise AnalysisError("invalid_contrast", "Contrast names must be short strings other than subject/contrast.")
        c.label(name)
        if not isinstance(weights, dict) or not 1 <= len(weights) <= len(terms) or set(weights)-set(terms):
            raise AnalysisError("invalid_contrast", "Each contrast must name only saved raw coefficient terms.")
        rows.append([c.c.check_number(weights.get(t, 0.0), "contrast weight", minimum=-1e12, maximum=1e12)
                     for t in terms])
        names.append(name)
        nulls.append(c.c.check_number(0.0 if null is None else null[name], "contrast null",
                                      minimum=-1e12, maximum=1e12))
    matrix = torch.tensor(rows, dtype=torch.float64)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if bool((norms == 0).any()):
        raise AnalysisError("unidentified_contrast", "A contrast cannot have all zero weights.")
    singular = torch.linalg.svdvals(matrix/norms[:, None])
    if float(singular[-1]) <= 1e-12*float(singular[0]):
        raise AnalysisError("unidentified_contrast", "Joint contrasts must be linearly independent.")
    return names, matrix, torch.tensor(nulls, dtype=torch.float64)


@resident_cpu
def conjoint_contrast(result, contrasts, *, null=None, level=0.95,
                      max_work=100_000_000, device="cpu", weights=None):
    """Saved raw-coefficient linear contrasts with full joint covariance and Wald tests.

    Contrasts is {name: {saved_raw_term: weight}}; omitted terms have weight zero.
    Null is an optional complete {name: value} map. Each saved respondent has
    pointwise method-specific t intervals. Iid Gaussian OLS uses joint F(q,n-p);
    HC/CR uses asymptotic chi-square(q), not an exact finite-sample robust test.
    Singular target covariance or redundant targets fail without pseudoinverses.
    """
    state = c.intact(result, "conjoint_fit")
    level = confidence(level)
    p, m = len(state["terms"]), len(state["subjects"])
    # Bound input size before constructing or checking the numerical target map.
    if not isinstance(contrasts, dict) or not 1 <= len(contrasts) <= p:
        raise AnalysisError("invalid_contrast", "Declare 1..p named linear contrasts.")
    q = len(contrasts)
    settings = c.guard("conjoint_contrast", q, p, m, device=device, weights=weights,
                       max_work=max_work, work=m*(p**3+q*p*p+q*q*p+q**3),
                       extra_buffers={"contrast_map_and_full_covariance": 8*(q*p+m*q*q)})
    names, mapping, null_values = contrast_spec(contrasts, null, state["terms"])
    transform = torch.tensor(state["transform"], dtype=torch.float64)
    # Calculate in the well-scaled fit basis to avoid raw-polynomial cancellation.
    scaled_mapping = mapping@transform
    rows, covariances, joint = [], [], []
    for record in state["subjects"]:
        beta = torch.tensor(record["parameters_scaled"], dtype=torch.float64)
        cov = torch.tensor(record["covariance_scaled"], dtype=torch.float64)
        estimate, target_cov = scaled_mapping@beta, scaled_mapping@cov@scaled_mapping.T
        target_cov = (target_cov+target_cov.T)/2
        if not torch.isfinite(estimate).all() or not torch.isfinite(target_cov).all():
            raise AnalysisError("numerical_failure", "Contrast estimates/covariance are nonfinite.")
        diagonal = torch.diagonal(target_cov)
        if bool((diagonal <= 0).any()):
            raise AnalysisError("singular_target_covariance", "Joint Wald inference needs positive target variances.")
        scale = torch.sqrt(diagonal)
        correlation = target_cov/scale[:, None]/scale[None, :]
        eigenvalues = torch.linalg.eigvalsh(correlation)
        if float(eigenvalues[0]) <= 1e-12*float(eigenvalues[-1]):
            raise AnalysisError("singular_target_covariance", "Joint target covariance is rank deficient; no pseudoinverse is used.")
        delta = (estimate-null_values)/scale
        wald = float(delta@torch.linalg.solve(correlation, delta))
        if not torch.isfinite(torch.tensor(wald, dtype=torch.float64)) or wald < -1e-10:
            raise AnalysisError("numerical_failure", "Joint Wald statistic is nonfinite or negative.")
        wald = max(0.0, wald)
        df = record["df"]
        classical = state["covariance"] == "nonrobust"
        statistic = wald/q if classical else wald
        probability = dist.f_sf(statistic, q, df) if classical else dist.chi2_sf(statistic, q)
        who = record["subject"]
        for i, name in enumerate(names):
            values = inference(float(estimate[i]), float(target_cov[i, i]), df, level)
            centered_infer = inference(float(estimate[i]-null_values[i]), float(target_cov[i, i]), df, level)
            rows.append([who, name, float(null_values[i]), *values[:3], centered_infer[3],
                         centered_infer[4], *values[5:]])
            covariances.append([who, name, *target_cov[i].tolist()])
        joint.append([who, "F" if classical else "chi2", statistic, q, df if classical else None, probability,
                      "exact conditional Gaussian iid" if classical else "asymptotic robust Wald"])
    return c.seal("conjoint_contrast", {
        "contrasts": table(rows, columns=["subject", "contrast", "null", "estimate", "std_error", "df",
                                          "t", "p_value", "ci_low", "ci_high"]),
        "covariance": table(covariances, columns=["subject", "contrast", *names]),
        "joint": table(joint, columns=["subject", "distribution", "statistic", "df1", "df2", "p_value", "inference"]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "terms": state["terms"],
        "contrasts": names, "matrix": mapping.tolist(), "null": null_values.tolist(),
        "covariance_method": state["covariance"], "level": level, "settings": settings},
        inference="pointwise t; joint exact Gaussian iid F or asymptotic robust chi2", **settings)
