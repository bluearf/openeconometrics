"""multivariate procedure family manifest (Torch-free).

Multivariate analysis in the tradition of SPSS ``FACTOR`` / ``RELIABILITY`` /
``QUICK CLUSTER`` / ``CLUSTER`` / ``DISCRIMINANT`` / ``CORRESPONDENCE`` and Stata
``pca``, ``factor``, ``alpha``, ``cluster``, ``discrim``, ``canon``, ``mds``, ``ca``:
principal components, factor analysis with rotations and scores, reliability,
k-means and hierarchical clustering, discriminant analysis, canonical
correlation, multidimensional scaling and correspondence analysis.

The family registers no estimator: every procedure is a descriptive analysis
that returns a ``core.TableSet`` of result tables whose ``attrs`` hold the
scalar results. Everything is computed in OpenEconometrics on float64 tensors
(one-pass moment matrices, ``torch.linalg`` eigen / QR / SVD decompositions).
See ``docs/econometrics/multivariate.md``.
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

_PACKAGE = "openecon.econometrics.multivariate"

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "pca": f"{_PACKAGE}.pca:pca",
    "pca_scores": f"{_PACKAGE}.pca:pca_scores",
    "pca_matrix": f"{_PACKAGE}.summary:pca_matrix",
    "pca_bootstrap": f"{_PACKAGE}.pca_uncertainty:pca_bootstrap",
    "pca_subspace_bootstrap": f"{_PACKAGE}.pca_subspace:pca_subspace_bootstrap",
    "pca_bootstrap_scores": f"{_PACKAGE}.pca_score_uncertainty:pca_bootstrap_scores",
    "pca_fweight_bootstrap": f"{_PACKAGE}.pca_frequency_uncertainty:pca_fweight_bootstrap",
    "pca_subspace_fweight_bootstrap": f"{_PACKAGE}.pca_frequency_uncertainty:pca_subspace_fweight_bootstrap",
    "factor": f"{_PACKAGE}.factor:factor",
    "factortest": f"{_PACKAGE}.factor:factortest",
    "factor_scores": f"{_PACKAGE}.factor:factor_scores",
    "factor_matrix": f"{_PACKAGE}.summary:factor_matrix",
    "factor_bootstrap": f"{_PACKAGE}.uncertainty:factor_bootstrap",
    "factor_multifactor_bootstrap": f"{_PACKAGE}.factor_uncertainty:factor_multifactor_bootstrap",
    "factor_multifactor_fweight_bootstrap": f"{_PACKAGE}.factor_frequency_uncertainty:factor_multifactor_fweight_bootstrap",
    "alpha": f"{_PACKAGE}.reliability:alpha",
    "cluster_kmeans": f"{_PACKAGE}.kmeans:cluster_kmeans",
    "cluster_assign": f"{_PACKAGE}.kmeans:cluster_assign",
    "cluster_hierarchical": f"{_PACKAGE}.hierarchical:cluster_hierarchical",
    "cluster_cut": f"{_PACKAGE}.hierarchical:cluster_cut",
    "discrim": f"{_PACKAGE}.discrim:discrim",
    "discrim_predict": f"{_PACKAGE}.discrim:discrim_predict",
    "discrim_summary": f"{_PACKAGE}.discrim:discrim_summary",
    "discrim_stepwise": f"{_PACKAGE}.discrim_selection:discrim_stepwise",
    "discrim_stepwise_predict": f"{_PACKAGE}.discrim_selection:discrim_stepwise_predict",
    "canon": f"{_PACKAGE}.canon:canon",
    "canon_matrix": f"{_PACKAGE}.canon:canon_matrix",
    "canon_scores": f"{_PACKAGE}.canon:canon_scores",
    "canon_bootstrap": f"{_PACKAGE}.canon_uncertainty:canon_bootstrap",
    "canon_fweight_bootstrap": f"{_PACKAGE}.canon_frequency_uncertainty:canon_fweight_bootstrap",
    "rm_anova_wide": f"{_PACKAGE}.repeated:rm_anova_wide",
    "rm_restore": f"{_PACKAGE}.repeated:rm_restore",
    "rm_contrasts": f"{_PACKAGE}.repeated:rm_contrasts",
    "mds": f"{_PACKAGE}.scaling:mds",
    "ca": f"{_PACKAGE}.scaling:ca",
    "ca_project": f"{_PACKAGE}.supplementary:ca_project",
    "ca_bootstrap": f"{_PACKAGE}.ca_uncertainty:ca_bootstrap",
}
