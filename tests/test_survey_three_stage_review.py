"""Malformed saved numeric dimensions refuse before scalar conversion callbacks."""

import pytest
from pydantic import ValidationError

from test_survey_three_stage_safety import METHODS, REGRESSION, declare, fixture, run


@pytest.fixture(scope="module")
def saved_states():
    frame = fixture()
    design = declare(frame)
    return {method: run(frame, design, method) for method in METHODS}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("defect", ["covariance-rows", "covariance-columns", "estimates", "null"])
def test_saved_dimensions_are_checked_before_pydantic_numeric_conversion(
    method, defect, saved_states
):
    state = saved_states[method]
    payload = state.model_dump(mode="python")
    payload["covariance"] = [list(row) for row in payload["covariance"]]
    conversions = []

    class NumericConversionProbe:
        """Pydantic StrictFloat accepts this protocol if nested parsing is reached."""

        def __float__(self):
            conversions.append("numeric conversion occurred")
            return 0.0

    if defect == "covariance-rows":
        payload["covariance"].append([NumericConversionProbe() for _ in state.labels])
    elif defect == "covariance-columns":
        payload["covariance"][0].append(NumericConversionProbe())
    else:
        key = (
            "null" if defect == "null" else "coefficients" if method in REGRESSION else "estimates"
        )
        payload[key] = [*payload[key], NumericConversionProbe()]

    with pytest.raises(ValidationError) as error:
        type(state).model_validate(payload)
    assert conversions == [], "Malformed numeric dimensions reached Pydantic tuple parsing."
    assert len(error.value.errors()) == 1
    assert error.value.errors()[0]["loc"] == (), "Admission should reject the entire raw state."
