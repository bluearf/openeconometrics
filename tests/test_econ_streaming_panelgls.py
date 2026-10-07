"""All current panel GLS/PCSE numerical option paths on ordered disk rows."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.core import make_spec
from openecon.econometrics.streaming_panelgls import fit_streaming_panelgls
from openecon.econometrics.systems.xtgls import fit_xtgls
from openecon.econometrics.systems.xtpcse import fit_xtpcse
from openecon.models import ResultBundle


def data(*, unbalanced=False, periods=37, groups=6):
    rng = np.random.default_rng(77621)
    shock = rng.normal(size=(periods, groups))*(.5+np.arange(groups)/groups)+rng.normal(size=(periods, 1))*.3
    error = np.zeros_like(shock)
    for time in range(periods):
        error[time] = shock[time]+(.3*error[time-1] if time else 0)
    frame = pd.DataFrame({"g": np.tile(np.arange(groups), periods), "t": np.repeat(np.arange(periods), groups),
                          "x": rng.normal(size=periods*groups), "category": np.tile(np.where(np.arange(groups)%2, "A", "B"), periods)})
    frame["y"] = 1+.6*frame.x+error.flatten()
    if unbalanced:
        frame = frame[~((frame.t%11==0)&(frame.g%2==0))]
    return frame.sample(frac=1, random_state=875).reset_index(drop=True)


def spec(estimator, *, panels="iid", corr="independent", rhotype="regress", igls=False,
         hetonly=False, independent=False, pairwise=False, np1=False, categorical=False):
    options = {"rhotype": rhotype}
    if estimator=="xtgls":
        options.update({"panels": panels, "corr": corr, "igls": igls})
    else:
        options.update({"correlation": corr, "hetonly": hetonly, "independent": independent, "pairwise": pairwise, "np1": np1})
    return make_spec(estimator, outcome="y", predictors=["x", *(["category"] if categorical else [])], panel="g", time="t",
                     categorical=["category"] if categorical else [], options=options)


def parity(a, b):
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=3e-8, atol=3e-10)
    for key, value in a.metrics.items():
        if value is None:
            assert b.metrics[key] is None
        else:
            assert b.metrics[key] == pytest.approx(value, rel=3e-8, abs=3e-9)
    for key in ["rho", "sigma", "sigma2", "sigma_matrix", "sigma_variances", "common_periods", "unpaired"]:
        if key in a.extra and not isinstance(a.extra[key], str):
            np.testing.assert_allclose(a.extra[key], b.extra[key], rtol=3e-8, atol=3e-9)
    assert a.nobs == b.nobs
    assert not b.sample_positions
    assert b.provenance["streaming"]["maximum_batch_rows"]<=13
    ResultBundle.model_validate_json(b.model_dump_json())


GLS = [(panels, "independent", "regress", False) for panels in ["iid", "heteroskedastic", "correlated"]]
GLS += [(panels, corr, rhotype, False) for panels in ["iid", "heteroskedastic", "correlated"] for corr in ["ar1", "psar1"] for rhotype in ["regress", "freg", "tscorr", "dw", "theil", "nagar"]]
GLS += [(panels, corr, "tscorr", True) for panels in ["iid", "heteroskedastic", "correlated"] for corr in ["independent", "ar1", "psar1"]]


@pytest.mark.parametrize("panels,corr,rhotype,igls", GLS)
def test_gls_all_current_panel_and_ar_methods(panels, corr, rhotype, igls):
    frame = data()
    s = spec("xtgls", panels=panels, corr=corr, rhotype=rhotype, igls=igls)
    parity(fit_xtgls(s, frame), fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13))


PCSE = [(corr, shape, "regress", False) for corr in ["independent", "ar1", "psar1"] for shape in ["iid", "hetonly", "casewise", "pairwise"]]
PCSE += [(corr, "casewise", rhotype, True) for corr in ["ar1", "psar1"] for rhotype in ["freg", "tscorr", "dw", "theil", "nagar"]]


@pytest.mark.parametrize("corr,shape,rhotype,np1", PCSE)
def test_pcse_all_covariance_shapes_and_ar_methods(corr, shape, rhotype, np1):
    frame = data(unbalanced=True)
    # Numeric missing periods genuinely violate the AR contract; datetime
    # inference ranks only globally observed dates, so use balanced AR samples.
    if corr!="independent":
        frame = data()
    s = spec("xtpcse", corr=corr, rhotype=rhotype, np1=np1, independent=shape=="iid", hetonly=shape=="hetonly", pairwise=shape=="pairwise")
    parity(fit_xtpcse(s, frame), fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13))


@pytest.mark.parametrize("estimator", ["xtgls", "xtpcse"])
def test_categories_global_pre_filter_mapping(estimator):
    frame = data()
    s = spec(estimator, panels="heteroskedastic", hetonly=True, categorical=True)
    reference = fit_xtgls(s, frame) if estimator=="xtgls" else fit_xtpcse(s, frame)
    parity(reference, fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13))
