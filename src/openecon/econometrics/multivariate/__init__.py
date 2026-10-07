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
    "factor": f"{_PACKAGE}.factor:factor",
    "factortest": f"{_PACKAGE}.factor:factortest",
    "factor_scores": f"{_PACKAGE}.factor:factor_scores",
    "alpha": f"{_PACKAGE}.reliability:alpha",
    "cluster_kmeans": f"{_PACKAGE}.kmeans:cluster_kmeans",
    "cluster_assign": f"{_PACKAGE}.kmeans:cluster_assign",
    "cluster_hierarchical": f"{_PACKAGE}.hierarchical:cluster_hierarchical",
    "cluster_cut": f"{_PACKAGE}.hierarchical:cluster_cut",
    "discrim": f"{_PACKAGE}.discrim:discrim",
    "discrim_predict": f"{_PACKAGE}.discrim:discrim_predict",
    "canon": f"{_PACKAGE}.canon:canon",
    "mds": f"{_PACKAGE}.scaling:mds",
    "ca": f"{_PACKAGE}.scaling:ca",
}
