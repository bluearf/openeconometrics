"""nonparametric family manifest (Torch-free).

Nonparametric and categorical-data procedures in the tradition of SPSS
``NPAR TESTS`` / ``CROSSTABS`` / ``ROC`` and Stata ``ranksum``, ``signrank``,
``kwallis``, ``ksmirnov``, ``swilk``, ``tabulate``: rank tests for independent
and related samples, one-sample distribution tests, cross-tabulation with
association measures and exact tests, and ROC analysis.

The family registers no estimator: every procedure is a test or a table and
returns ``core.table`` / ``core.TableSet`` results whose ``attrs`` hold the
statistics, p-values and settings. Everything is computed in OpenEconometrics on
float64 tensors (sorting-based ranks, exact null distributions by dynamic
programming, published approximations for Shapiro-Wilk and Lilliefors). See
``docs/econometrics/nonparametric.md``.
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

_PACKAGE = "openecon.econometrics.nonparametric"

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "ranksum": f"{_PACKAGE}.independent:ranksum",
    "kwallis": f"{_PACKAGE}.independent:kwallis",
    "median_test": f"{_PACKAGE}.independent:median_test",
    "jonckheere": f"{_PACKAGE}.independent:jonckheere",
    "signrank": f"{_PACKAGE}.paired:signrank",
    "signtest": f"{_PACKAGE}.paired:signtest",
    "mcnemar": f"{_PACKAGE}.paired:mcnemar",
    "symmetry": f"{_PACKAGE}.paired:symmetry",
    "friedman": f"{_PACKAGE}.paired:friedman",
    "cochran_q": f"{_PACKAGE}.paired:cochran_q",
    "ksmirnov": f"{_PACKAGE}.distribution:ksmirnov",
    "runtest": f"{_PACKAGE}.distribution:runtest",
    "bitest": f"{_PACKAGE}.distribution:bitest",
    "prtest": f"{_PACKAGE}.distribution:prtest",
    "chi2gof": f"{_PACKAGE}.distribution:chi2gof",
    "swilk": f"{_PACKAGE}.normality:swilk",
    "sfrancia": f"{_PACKAGE}.normality:sfrancia",
    "sktest": f"{_PACKAGE}.normality:sktest",
    "crosstab": f"{_PACKAGE}.crosstab:crosstab",
    "tabulate": f"{_PACKAGE}.crosstab:tabulate",
    "roc": f"{_PACKAGE}.roc:roc",
    "roccomp": f"{_PACKAGE}.roc:roccomp",
}
