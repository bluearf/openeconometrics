"""Public transport export cannot hide an invalid complete saved model."""

import inspect

import pandas as pd
import pytest
from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.supervised import cart as module
from openecon.econometrics.supervised.cart import CartQueryState, CartState, cart, cart_predict
from openecon.econometrics.supervised.split import SplitState, prediction_split


@pytest.mark.parametrize("cls", [SplitState, CartState, CartQueryState])
@pytest.mark.parametrize("kwargs", [{"exclude": {"payload"}}, {"include": set()}])
def test_partial_export_checks_complete_invalid_state(cls, kwargs):
    forged = cls.model_construct(payload={"bad": True})
    with pytest.raises(AnalysisError):
        forged.model_dump(**kwargs)


def test_valid_partial_exports_replay_without_growing(monkeypatch):
    frame = pd.DataFrame({"x": list(range(16)), "y": [0.0] * 8 + [1.0] * 8})
    split = prediction_split(frame, roles=["train"] * 12 + ["test"] * 4)
    fitted = cart(frame, outcome="y", features=["x"], split=split, min_leaf=2)
    query = cart_predict(fitted, frame)
    states = [split, CartState.model_validate({"payload": fitted.attrs["state"]}),
              CartQueryState.model_validate({"payload": query.attrs})]

    def forbid(*args, **kwargs):
        raise AssertionError("Export must not grow/refit a tree")

    monkeypatch.setattr(module, "_grow", forbid)
    for state in states:
        assert state.model_dump(exclude={"payload"}) == {}
        assert state.model_dump(include=set()) == {}
        assert inspect.signature(type(state).model_dump) == inspect.signature(BaseModel.model_dump)
