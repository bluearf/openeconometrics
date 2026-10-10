"""Scalar instrumental quantile profiles with declared finite search domains."""

ESTIMATORS = ()
EXPORTS = {
    "ivqreg": "openecon.econometrics.ivquantile.api:ivqreg",
    "ivqreg_restore": "openecon.econometrics.ivquantile.api:ivqreg_restore",
    "ivqreg_predict": "openecon.econometrics.ivquantile.api:ivqreg_predict",
}

__all__ = ["ivqreg", "ivqreg_predict", "ivqreg_restore"]


def __getattr__(name):
    if name in EXPORTS:
        from importlib import import_module
        return getattr(import_module("openecon.econometrics.ivquantile.api"), name)
    raise AttributeError(name)
