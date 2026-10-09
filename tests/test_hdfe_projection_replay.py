"""Cached disk projection preserves the former source-replay numerical path."""

import struct

import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import streaming_hdfe as hdfe, streaming_ppml as ppml
from test_econ_streaming_hdfe import data as hdfe_data, spec as hdfe_spec
from test_econ_streaming_ppml import fixture as ppml_data, spec as ppml_spec


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


class Uncached:
    """Run the unchanged original projection equations against source batches."""

    def __init__(self, sample, *_):
        self.sample = sample

    def __getattr__(self, name):
        return getattr(self.sample, name)

    def diagnostics(self):
        return {}


def case(model, covariance="robust", weight=None):
    if model == "ppmlhdfe":
        return (
            ppml_data(n=180),
            ppml_spec(("a", "b", "c"), covariance, weight, categorical=True),
            ppml.fit_streaming,
        )
    return (
        hdfe_data(n=181),
        hdfe_spec(
            3,
            covariance,
            cluster=["cluster", "cluster2"] if covariance == "cluster" else None,
            weight_type=weight,
            categorical=True,
        ),
        hdfe.fit_streaming_hdfe,
    )


@pytest.mark.parametrize("model", ["ppmlhdfe", "reghdfe"])
@pytest.mark.parametrize(
    "covariance,weight",
    [("nonrobust", None), ("robust", "fweight"), ("robust", "aweight"), ("cluster", "pweight")],
)
def test_identical_equations_weights_sample_and_inference_with_fewer_source_passes(
    monkeypatch, model, covariance, weight
):
    frame, spec, fit = case(model, covariance, weight)
    cached = fit(spec, Dataset.from_frame(frame), batch_rows=47)
    monkeypatch.setattr(hdfe, "_ProjectionReplay", Uncached)
    monkeypatch.setattr(ppml, "_ProjectionReplay", Uncached)
    reference = fit(spec, Dataset.from_frame(frame), batch_rows=47)
    assert cached.coefficients == reference.coefficients
    assert cached.covariance_matrix == reference.covariance_matrix
    assert cached.metrics == reference.metrics
    assert cached.inference == reference.inference
    assert cached.tests == reference.tests
    assert cached.predictions == reference.predictions
    assert cached.nobs == reference.nobs and cached.dropped_rows == reference.dropped_rows
    assert cached.provenance["data_hash"] == reference.provenance["data_hash"]
    assert cached.provenance["streaming"]["passes"] < reference.provenance["streaming"]["passes"]
    assert cached.provenance["solver_diagnostics"]["FE_geometry_maximum_rows"] <= 47
    assert cached.provenance["solver_diagnostics"]["FE_geometry_replays"] > 0


@pytest.mark.parametrize("model", ["ppmlhdfe", "reghdfe"])
def test_source_content_change_during_cached_projection_still_refused(monkeypatch, model):
    frame, spec, fit = case(model)
    source = Dataset.from_batches(lambda: iter([frame]), columns=list(frame), row_count=len(frame))
    original = hdfe._project
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            frame.loc[50, "x"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hdfe, "_project", mutate)
    with pytest.raises(AnalysisError, match="changed"):
        fit(spec, source, batch_rows=47)


@pytest.mark.parametrize("damage", ["weights", "oversized_header", "partial_header"])
def test_owned_geometry_corruption_refused_before_projection(monkeypatch, tmp_path, damage):
    frame, spec, fit = case("reghdfe")
    original = hdfe._ProjectionReplay.batches
    changed = False

    def broken(self):
        nonlocal changed
        if not changed:
            changed = True
            if damage == "weights":
                self.states.file("projection_weights").write_bytes(b"bad")
            elif damage == "oversized_header":
                self.states.file("projection_rows").write_bytes(struct.pack("<QQ", 2**64 - 1, 0))
            else:
                self.states.file("projection_rows").write_bytes(b"partial")
        yield from original(self)

    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(hdfe._ProjectionReplay, "batches", broken)
    with pytest.raises(AnalysisError, match="Owned FE"):
        fit(spec, Dataset.from_frame(frame), batch_rows=47)
    assert not list(tmp_path.iterdir())
