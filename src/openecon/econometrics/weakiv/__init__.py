"""Whole-real-line scalar iid conditional likelihood ratio inference."""

ESTIMATORS = ()
EXPORTS = {
    "iv_clr_confidence_set": "openecon.econometrics.weakiv.api:iv_clr_confidence_set",
    "iv_clr_restore": "openecon.econometrics.weakiv.api:iv_clr_restore",
    "iv_clr_test": "openecon.econometrics.weakiv.api:iv_clr_test",
    "iv_clr_test_restore": "openecon.econometrics.weakiv.api:iv_clr_test_restore",
}

__all__ = [
    "CLRConfidenceSet",
    "CLRTestState",
    "iv_clr_confidence_set",
    "iv_clr_restore",
    "iv_clr_test",
    "iv_clr_test_restore",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module("openecon.econometrics.weakiv.api"), name)
    globals()[name] = value
    return value
