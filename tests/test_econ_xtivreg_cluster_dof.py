"""Independent raw-coordinate CGM oracle for any-cluster FE nesting."""
import numpy as np
import pandas as pd
from types import SimpleNamespace

from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.models import ModelSpec


def test_private_any_cluster_panel_nesting_covariance_order_and_numpy_oracle(monkeypatch):
    # Capture the private numerical branch before ResultBundle revalidates the
    # deliberately wider private spec. Public xtivreg continues to accept one
    # cluster dimension, so this fixture cannot create a persisted public result.
    def capture_result(frame, **values):
        return SimpleNamespace(covariance_matrix=values["covariance"].tolist(),
                               coefficients=[SimpleNamespace(estimate=float(value))
                                             for value in values["params"]])
    monkeypatch.setattr("openecon.econometrics.iv.xtivreg.build_result", capture_result)
    rng = np.random.default_rng(581133)
    n = 483
    panel = np.arange(n)%23
    x, z, v, noise = rng.normal(size=(4, n))
    endog = .4*x+.7*z+v+.03*panel
    data = pd.DataFrame({"panel": panel, "other": np.arange(n)%17,
                         "x": x+.7, "z": z-.3, "endog": endog,
                         "y": 1.1*(x+.7)-.4*endog+.08*panel+noise+.3*v})
    base = ModelSpec(estimator="xtivreg", outcome="y", predictors=["x"], panel="panel",
                     columns={"endogenous": ["endog"], "instruments": ["z"]},
                     covariance="cluster", cluster="other", options={"model": "fe", "small": True})
    # The public xtivreg manifest currently accepts ONE cluster dimension.
    # Exercise the helper defensively through an explicit private branch; this
    # test does not add two-way clustering to the public specification/API.
    base = base.model_copy(update={"cluster": ["other", "panel"]})
    first = fit_xtivreg(base, data)
    reverse = fit_xtivreg(base.model_copy(update={"cluster": ["panel", "other"]}), data)
    np.testing.assert_allclose(first.covariance_matrix, reverse.covariance_matrix, rtol=3e-11, atol=3e-12)
    raw_x = data[["x", "endog"]].to_numpy()
    raw_z = data[["x", "z"]].to_numpy()
    y = data.y.to_numpy()
    xw, zw, yw = raw_x.copy(), raw_z.copy(), y.copy()
    for group in np.unique(panel):
        selected = panel == group
        xw[selected] -= raw_x[selected].mean(0)
        zw[selected] -= raw_z[selected].mean(0)
        yw[selected] -= y[selected].mean()
    xg = np.column_stack([np.ones(n), xw+raw_x.mean(0)])
    zg = np.column_stack([np.ones(n), zw+raw_z.mean(0)])
    projected = zg@np.linalg.lstsq(zg, xg, rcond=None)[0]
    bread = np.linalg.inv(projected.T@projected)
    beta = bread@(projected.T@(yw+y.mean()))
    residual = yw+y.mean()-xg@beta
    scores = projected*residual[:, None]
    meat = np.zeros((3, 3))
    for names, sign in [(["other"], 1), (["panel"], 1), (["other", "panel"], -1)]:
        labels = [tuple(row) for row in data[names].to_numpy()]
        groups = sorted(set(labels))
        sums = np.stack([scores[np.array([label == group for label in labels])].sum(0) for group in groups])
        meat += sign*(sums.T@sums)
    eigenvalues, vectors = np.linalg.eigh((meat+meat.T)/2)
    meat = (vectors*np.maximum(eigenvalues, 0))@vectors.T
    # This engine uses G_min for one common finite-sample correction after CGM
    # inclusion/exclusion and clipping, rather than per-dimension corrections.
    expected = bread@meat@bread*17/16*(n-1)/(n-3)
    np.testing.assert_allclose(first.covariance_matrix, expected, rtol=3e-9, atol=3e-11)
    np.testing.assert_allclose([c.estimate for c in first.coefficients], beta, rtol=3e-11, atol=3e-12)
