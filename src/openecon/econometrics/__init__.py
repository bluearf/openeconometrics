"""OpenEconometrics's econometric estimator families.

Every estimator here is implemented in OpenEconometrics on float64 PyTorch tensors;
no third-party estimation library runs at fit time. Importing this package is
cheap: estimator code loads only when a model is fitted. See
``openecon.econometrics.registry`` for the catalogue and
``openecon.econometrics.core`` for the shared sample, design and result layer.
"""
