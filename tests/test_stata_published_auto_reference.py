"""Native results against rounded primary-source Stata publications.

No Stata process, external estimator or reconstructed full covariance oracle is
used. Decimal strings retain the precision actually published. The historical
mfx reference validates continuous MEM point estimates, not AME or its SE/CI.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import pytest

import openecon as oe
from openecon.dataset import Dataset
from openecon.models import ResultBundle


FIXTURES = Path(__file__).parent / "fixtures" / "stata"
REFERENCE = json.loads((FIXTURES / "published-auto-reference.json").read_text())
COEFFICIENT_FIELDS = ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")


def rounding_tolerance(printed: str, observed: float) -> Decimal:
    """Half the last displayed unit, plus insignificant binary64 readback noise."""
    value = Decimal(printed)
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    roundoff = Decimal(str(32 * sys.float_info.epsilon * max(1, abs(float(value)), abs(observed))))
    return quantum / 2 + roundoff


def assert_rounded(observed: float, printed: str) -> None:
    assert isinstance(printed, str), "Reference precision must be retained as printed text"
    observed = float(observed)
    assert math.isfinite(observed)
    error = abs(Decimal(str(observed)) - Decimal(printed))
    allowed = rounding_tolerance(printed, observed)
    assert error <= allowed, f"Observed {observed:.17g}, printed {printed}: error {error} > {allowed}"


def reference_cells(case: str):
    """Yield only printed decimal cells; counts and degrees of freedom are exact."""
    ref = REFERENCE[case]
    if case in {"ols", "logit"}:
        for term, row in ref["coefficients"].items():
            for field in COEFFICIENT_FIELDS:
                yield f"coefficients.{term}.{field}", row[field]
        for field, value in ref["metrics"].items():
            yield f"metrics.{field}", value
        for field, value in ref.get("model_test", {}).items():
            yield f"model_test.{field}", value
    elif case == "ttest":
        for table in ("statistics", "test"):
            for field, value in ref[table].items():
                yield f"{table}.{field}", value
    else:
        for table in ("means", "effect_table", "hand_display"):
            for field, value in ref[table].items():
                yield f"{table}.{field}", value
        for field in ("probability_at_means", "link_derivative_at_means"):
            yield field, ref[field]


def model_observations(result, case: str) -> dict[str, float]:
    observations = {}
    for coefficient in result.coefficients:
        for field in COEFFICIENT_FIELDS:
            observations[f"coefficients.{coefficient.term}.{field}"] = float(getattr(coefficient, field))
    for field in REFERENCE[case]["metrics"]:
        observations[f"metrics.{field}"] = float(result.metrics[field])
    if case == "ols":
        for field in REFERENCE[case]["model_test"]:
            observations[f"model_test.{field}"] = float(result.tests["model"][field])
    return observations


def ttest_observations(result) -> dict[str, float]:
    return {f"{table}.{field}": float(result[table].loc[label, field])
            for table, label in (("statistics", "mpg"), ("test", "one_sample"))
            for field in REFERENCE["ttest"][table]}


def mem_observations(result, data, frame) -> dict[str, float]:
    """Evaluate saved-state effects without outcomes or saved prediction samples."""
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    output = oe.margins(restored, data=data, variables=["weight", "length"], method="mem").set_index("variable")
    means = frame[["weight", "length"]].mean().to_frame().T
    probability = float(oe.predict(restored, means)["response"].iloc[0])
    observations = {"probability_at_means": probability,
                    "link_derivative_at_means": probability * (1 - probability)}
    for variable in ("weight", "length"):
        observations[f"means.{variable}"] = float(means[variable].iloc[0])
        for table in ("effect_table", "hand_display"):
            observations[f"{table}.{variable}"] = float(output.loc[variable, "estimate"])
    return observations


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(FIXTURES / "auto-numeric.csv")


@pytest.fixture(scope="module", params=["resident", "replay_seven_rows"])
def native(request, frame):
    audit = {"producer_passes": 0, "producer_peak_rows": 0}

    def factory():
        audit["producer_passes"] += 1
        for start in range(0, len(frame), 7):
            block = frame.iloc[start:start + 7]
            audit["producer_peak_rows"] = max(audit["producer_peak_rows"], len(block))
            yield block

    data = (frame if request.param == "resident" else
            Dataset.from_batches(factory, frame.columns.tolist(), row_count=len(frame)))
    ols = oe.ols(data=data, y="mpg", x=["weight", "foreign"],
                 covariance="nonrobust", device="cpu")
    ttest = oe.ttest(data, "mpg", mu=20)
    logit = oe.logit(data=data, y="foreign", x=["weight", "length"], covariance="nonrobust")
    return {"data": data, "frame": frame, "path": request.param,
            "audit": audit, "ols": ols, "ttest": ttest, "logit": logit}


def check_cells(case, observations):
    cells = list(reference_cells(case))
    assert cells
    for path, printed in cells:
        assert_rounded(observations[path], printed)


def test_fixture_is_exact_pinned_numeric_projection(frame):
    provenance = json.loads((FIXTURES / "auto-source.json").read_text())
    assert hashlib.sha256((FIXTURES / "auto-numeric.csv").read_bytes()).hexdigest() == provenance["csv_sha256"]
    assert len(frame) == provenance["rows"] == 74
    assert frame.columns.tolist() == provenance["numeric_projection"]
    assert frame.notna().all().all()
    assert set(frame.foreign) == {0, 1}
    assert provenance["source_url"] == "https://www.stata-press.com/data/r19/auto.dta"
    assert REFERENCE["fresh_stata_execution"] is False
    assert REFERENCE["mem"]["standard_errors_published"] is False
    assert REFERENCE["mem"]["confidence_intervals_published"] is False


def test_ols_published_coefficients_uncertainty_and_model_summary(native):
    fit = native["ols"]
    assert fit.nobs == REFERENCE["ols"]["exact"]["nobs"]
    assert fit.nobs_original == 74 and fit.dropped_rows == 0
    assert fit.metrics["df_model"] == 2 and fit.metrics["df_resid"] == 71
    assert fit.inference["use_t"] is True
    assert fit.inference["covariance"] == "nonrobust"
    check_cells("ols", model_observations(fit, "ols"))
    assert "\\begin{tabular}" in fit.to_latex()


def test_ttest_published_mean_intervals_and_all_probability_tails(native):
    fit = native["ttest"]
    assert fit["statistics"].loc["mpg", "n"] == 74
    assert fit["test"].loc["one_sample", "df"] == 73
    check_cells("ttest", ttest_observations(fit))
    # Do not accidentally validate a difference interval as a mean interval.
    for bound in ("ci_low", "ci_high"):
        mean_bound = float(fit["statistics"].loc["mpg", bound])
        difference_bound = float(fit["test"].loc["one_sample", bound])
        assert abs(mean_bound - difference_bound - 20) < 1e-13
    assert "\\begin{tabular}" in fit.to_latex()


def test_logit_published_coefficients_uncertainty_and_likelihood(native):
    restored = ResultBundle.model_validate_json(native["logit"].model_dump_json())
    assert restored.nobs == 74 and restored.nobs_original == 74 and restored.dropped_rows == 0
    assert restored.inference["use_t"] is False
    assert restored.inference["covariance"] == "nonrobust"
    check_cells("logit", model_observations(restored, "logit"))
    assert "\\begin{tabular}" in restored.to_latex()


def test_json_restored_mem_published_probability_link_slope_and_continuous_effects(native):
    check_cells("mem", mem_observations(native["logit"], native["data"], native["frame"]))


def test_historical_mfx_protocol_is_mem_not_ame(native):
    restored = ResultBundle.model_validate_json(native["logit"].model_dump_json())
    ame = oe.margins(restored, data=native["data"], variables=["weight", "length"], method="ame").set_index("variable")
    mem = oe.margins(restored, data=native["data"], variables=["weight", "length"], method="mem").set_index("variable")
    for variable in ("weight", "length"):
        assert abs(float(ame.loc[variable, "estimate"] - mem.loc[variable, "estimate"])) > 1e-5


def test_replay_factory_actually_consumed_multiple_seven_row_blocks(native):
    if native["path"] == "resident":
        assert native["audit"] == {"producer_passes": 0, "producer_peak_rows": 0}
    else:
        assert native["audit"]["producer_passes"] >= 3
        assert native["audit"]["producer_peak_rows"] == 7
        assert native["ols"].provenance["streaming"]["maximum_encoded_rows"] <= 7
        assert native["logit"].provenance["streaming"]["maximum_encoded_batch_rows"] <= 7


def test_printed_digit_tolerance_preserves_trailing_zero_precision():
    assert rounding_tolerance("0.130", .13) < Decimal("0.000500000001")
    assert rounding_tolerance("0.130", .13) < rounding_tolerance("0.13", .13)
    assert_rounded(.0006371 + 4.9e-8, ".0006371")
    with pytest.raises(AssertionError):
        assert_rounded(.0006371 + 1e-7, ".0006371")
    with pytest.raises(AssertionError):
        assert_rounded(float("nan"), ".0006371")
