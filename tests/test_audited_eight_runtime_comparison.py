"""Adversarial proof boundaries for the complete source/installed comparator."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from openecon.models import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audited_runtime_comparison", ROOT/"scripts/compare_audited_eight_results.py")
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


def uuid(number):
    return f"{number:08x}-0000-4000-8000-{number:012x}"


def bundle(number):
    return {
        "id": uuid(number), "created_at": f"2026-10-07T12:00:{number:02d}+00:00",
        "spec": ModelSpec(estimator="ols", outcome="y", predictors=[]).model_dump(mode="json"),
        "nobs": 3, "nobs_original": 3, "dropped_rows": 0, "sample_positions": [0, 1, 2],
        "coefficients": [{"term": "Intercept", "estimate": 1.0, "std_error": 0.2,
                          "statistic": 5.0, "p_value": 0.02, "ci_low": 0.6, "ci_high": 1.4}],
        "covariance_matrix": [[0.04]], "metrics": {"rmse": 0.3},
        "warnings": [], "predictions": [], "provenance": {}, "inference": {"use_t": True},
        "title": "Scientific title", "tests": {}, "extra": {},
    }


def summary(attrs=None):
    return {"schema": "openecon.summary.v1", "title": "Full output", "attrs": attrs or {},
            "tables": {"values": {"columns": ["effect"], "data": [[1.0]], "index": [0]}}}


def receipts(tmp_path, left, right):
    paths = []
    for name, files in (("source", left), ("installed", right)):
        folder = tmp_path/name
        folder.mkdir()
        roster = {}
        for filename, value in files.items():
            path = folder/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = value if isinstance(value, bytes) else value.encode() if isinstance(value, str) else json.dumps(value).encode()
            path.write_bytes(raw)
            roster[filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        receipt = tmp_path/f"{name}-receipt.json"
        receipt.write_text(json.dumps({"output_directory": str(folder), "files": roster}))
        paths.append(receipt)
    return paths


def run(tmp_path, left, right):
    paths = receipts(tmp_path, left, right)
    return comparator.compare(*paths, tmp_path/"comparison.json", tmp_path/"raw.tar.gz")


def test_valid_bundle_provenance_and_actual_file_reference_preserve_both_raw_sets(tmp_path):
    left, right = bundle(1), bundle(2)
    files = []
    for saved in (left, right):
        raw_hash = hashlib.sha256(json.dumps(saved).encode()).hexdigest()
        files.append({"model.json": saved, "mean.json": summary({"source_result_id": saved["id"]}),
                      "example-receipt.json": {"files": {"model.json": raw_hash}},
                      "mean.tex": "Exact scientific result & 1.0 \\\\\n"})
    report = run(tmp_path, *files)
    assert report["complete_files"] == 4
    assert report["validated_result_bundles"] == 1
    assert report["exact_scientific_content_equal"]
    assert {row["path"] for row in report["normalized_paths"]} == {
        "model.json/id", "model.json/created_at", "mean.json/attrs/source_result_id",
        "example-receipt.json/files/model.json",
    }
    with tarfile.open(tmp_path/"raw.tar.gz") as archive:
        assert len(archive.getnames()) == 10
        for label, saved in (("source", left), ("installed", right)):
            assert json.loads(archive.extractfile(f"{label}/model.json").read()) == saved


@pytest.mark.parametrize("field,value", [
    ("category_label", lambda b: b["id"]),
    ("observation_time", lambda b: b["created_at"]),
    ("embedded_label", lambda b: f"literal scientific {b['id']} substring"),
])
def test_scientific_strings_equal_provenance_tokens_are_never_normalized(tmp_path, field, value):
    first, second = bundle(1), bundle(2)
    first["extra"][field], second["extra"][field] = value(first), value(second)
    with pytest.raises(comparator.ComparisonError, match=f"extra/{field}"):
        run(tmp_path, {"model.json": first}, {"model.json": second})


def test_scientific_coefficient_label_can_equal_uuid_without_becoming_provenance(tmp_path):
    first, second = bundle(1), bundle(2)
    first["coefficients"][0]["term"], second["coefficients"][0]["term"] = first["id"], second["id"]
    with pytest.raises(comparator.ComparisonError, match="coefficients/0/term"):
        run(tmp_path, {"model.json": first}, {"model.json": second})


@pytest.mark.parametrize("token", ["id", "created_at"])
def test_latex_scientific_substrings_are_compared_exactly(tmp_path, token):
    first, second = bundle(1), bundle(2)
    with pytest.raises(comparator.ComparisonError, match="no substring normalization"):
        run(tmp_path, {"model.json": first, "result.tex": f"scientific prefix {first[token]} suffix"},
            {"model.json": second, "result.tex": f"scientific prefix {second[token]} suffix"})


def test_partial_bundle_lookalike_does_not_authorize_id_mapping(tmp_path):
    first, second = [dict(spec={"estimator": "ols"}, coefficients=[], created_at=bundle(n)["created_at"],
                          id=uuid(n)) for n in (1, 2)]
    with pytest.raises(comparator.ComparisonError, match="Invalid typed ResultBundle"):
        run(tmp_path, {"fake.json": first}, {"fake.json": second})


@pytest.mark.parametrize("field,value", [("id", "arbitrary scientific string"), ("created_at", "2026-10-07T12:00:00")])
def test_untyped_uuid_or_timezone_free_creation_time_is_rejected(tmp_path, field, value):
    first, second = bundle(1), bundle(2)
    first[field] = value
    with pytest.raises(comparator.ComparisonError, match="Invalid typed ResultBundle"):
        run(tmp_path, {"model.json": first}, {"model.json": second})


@pytest.mark.parametrize("field,reverse", [("id", False), ("id", True), ("created_at", False), ("created_at", True)])
def test_ambiguous_identity_or_timestamp_correspondence_is_rejected(tmp_path, field, reverse):
    first = {"one.json": bundle(1), "two.json": bundle(2)}
    second = {"one.json": bundle(3), "two.json": bundle(4)}
    values = first if reverse else second
    values["two.json"][field] = values["one.json"][field]
    with pytest.raises(comparator.ComparisonError, match="Ambiguous"):
        run(tmp_path, first, second)


def test_repeated_full_bundle_uuid_cannot_anchor_two_different_scientific_payloads(tmp_path):
    first, second = bundle(1), bundle(2)
    other_first, other_second = deepcopy(first), deepcopy(second)
    other_first["coefficients"][0]["estimate"] = other_second["coefficients"][0]["estimate"] = 7.0
    with pytest.raises(comparator.ComparisonError, match="bundle UUID anchor"):
        run(tmp_path, {"one.json": first, "two.json": other_first},
            {"one.json": second, "two.json": other_second})


def test_exact_duplicate_full_bundles_are_valid_references_to_one_fit(tmp_path):
    first, second = bundle(1), bundle(2)
    report = run(tmp_path, {"one.json": first, "copy.json": deepcopy(first)},
                 {"one.json": second, "copy.json": deepcopy(second)})
    assert report["validated_result_bundles"] == 2 and report["exact_scientific_content_equal"]


def origin(number):
    saved = bundle(number)
    return {"result_id": saved["id"], "spec": saved["spec"], "coefficients": saved["coefficients"],
            "covariance_matrix": saved["covariance_matrix"], "inference": saved["inference"],
            "metrics": saved["metrics"], "nobs": 3, "sample_positions": [5, 6, 7],
            "input_positions": [5, 6, 7], "origin": 7, "status": "ok"}


def test_complete_rolling_scientific_origin_authorizes_only_its_result_id(tmp_path):
    first, second = summary({"origins": [origin(1)]}), summary({"origins": [origin(2)]})
    report = run(tmp_path, {"rolling.json": first}, {"rolling.json": second})
    assert report["validated_rolling_origins"] == 1
    assert [row["path"] for row in report["normalized_paths"]] == ["rolling.json/attrs/origins/0/result_id"]


def test_repeated_origin_uuid_cannot_anchor_divergent_scientific_records(tmp_path):
    first, second = origin(1), origin(2)
    other_first, other_second = deepcopy(first), deepcopy(second)
    other_first["coefficients"][0]["estimate"] = other_second["coefficients"][0]["estimate"] = 7.0
    with pytest.raises(comparator.ComparisonError, match="origin UUID anchor"):
        run(tmp_path, {"rolling.json": summary({"origins": [first, other_first]})},
            {"rolling.json": summary({"origins": [second, other_second]})})


def test_exact_duplicate_full_origin_records_are_permitted(tmp_path):
    first, second = origin(1), origin(2)
    report = run(tmp_path, {"rolling.json": summary({"origins": [first, deepcopy(first)]})},
                 {"rolling.json": summary({"origins": [second, deepcopy(second)]})})
    assert report["validated_rolling_origins"] == 2 and report["exact_scientific_content_equal"]


@pytest.mark.parametrize("bundle_first", [False, True])
def test_full_bundle_and_origin_cannot_share_a_uuid_anchor(tmp_path, bundle_first):
    first, second = bundle(1), bundle(2)
    one, two = origin(1), origin(2)
    one["coefficients"][0]["estimate"] = two["coefficients"][0]["estimate"] = 7.0
    bundle_name, origin_name = ("a-bundle.json", "b-origin.json") if bundle_first else ("b-bundle.json", "a-origin.json")
    with pytest.raises(comparator.ComparisonError, match="cross-kind UUID alias"):
        run(tmp_path, {bundle_name: first, origin_name: summary({"origins": [one]})},
            {bundle_name: second, origin_name: summary({"origins": [two]})})


@pytest.mark.parametrize("damage", ["missing_covariance", "bad_covariance_geometry", "changed_estimate"])
def test_rolling_origin_keeps_complete_science_and_refuses_partial_records(tmp_path, damage):
    first, second = origin(1), origin(2)
    if damage == "missing_covariance":
        first.pop("covariance_matrix")
        second.pop("covariance_matrix")
    elif damage == "bad_covariance_geometry":
        second["covariance_matrix"] = []
    else:
        second["coefficients"][0]["estimate"] += 0.01
    with pytest.raises(comparator.ComparisonError):
        run(tmp_path, {"rolling.json": summary({"origins": [first]})},
            {"rolling.json": summary({"origins": [second]})})


def test_unrecognized_reference_field_name_does_not_authorize_mapping(tmp_path):
    first, second = bundle(1), bundle(2)
    with pytest.raises(comparator.ComparisonError, match="Unrecognized or scientific"):
        run(tmp_path, {"model.json": first, "science.json": {"source_result_id": first["id"]}},
            {"model.json": second, "science.json": {"source_result_id": second["id"]}})


def sspace():
    import openecon as oe

    return oe.sspace(data={"y": [0.2, 0.1, 0.4]}, y="y",
                     system={"T": [[0.5]], "Z": [[1.0]], "Q": [[0.2]], "H": [[0.3]]}).model_dump(mode="json")


def with_digest(saved, number):
    value = deepcopy(saved)
    value.update(id=uuid(number), created_at=bundle(number)["created_at"])
    value["extra"].pop("state_sha256", None)
    value["extra"]["state_sha256"] = comparator.digest(value)
    return value


def test_sspace_full_digest_and_summary_references_are_independently_verified(tmp_path):
    model = sspace()
    first, second = with_digest(model, 1), with_digest(model, 2)
    report = run(tmp_path, {"model.json": first, "smooth.json": summary({"state_sha256": first["extra"]["state_sha256"]})},
                 {"model.json": second, "smooth.json": summary({"state_sha256": second["extra"]["state_sha256"]})})
    assert report["exact_scientific_content_equal"]
    assert any(row["path"] == "model.json/extra/state_sha256" for row in report["normalized_paths"])


@pytest.mark.parametrize("repair_digest", [False, True])
def test_full_sspace_science_change_is_refused_even_with_a_valid_recomputed_digest(tmp_path, repair_digest):
    model = sspace()
    first, second = with_digest(model, 1), with_digest(model, 2)
    second["extra"]["next_covariance"][0][0] += 0.1
    if repair_digest:
        second = with_digest(second, 2)
    with pytest.raises(comparator.ComparisonError, match="digest differs|next_covariance"):
        run(tmp_path, {"model.json": first}, {"model.json": second})


def test_reference_sha_must_be_the_actual_referenced_file_hash_even_if_equal(tmp_path):
    with pytest.raises(comparator.ComparisonError, match="SHA differs from actual file"):
        run(tmp_path, {"model.json": bundle(1), "example-receipt.json": {"files": {"model.json": "0"*64}}},
            {"model.json": bundle(2), "example-receipt.json": {"files": {"model.json": "0"*64}}})


def test_actual_file_hash_mismatch_and_unlisted_file_are_refused(tmp_path):
    first, second = receipts(tmp_path, {"x.json": {"x": 1}}, {"x.json": {"x": 1}})
    (tmp_path/"installed"/"x.json").write_text('{"x":2}')
    with pytest.raises(comparator.ComparisonError, match="Actual artifact SHA differs"):
        comparator.Comparison(first, second)
    (tmp_path/"installed"/"extra.tex").write_text("unlisted full output")
    with pytest.raises(comparator.ComparisonError, match="complete receipt file roster"):
        comparator.Comparison(first, second)


def mi_pool(ids, *, label="same scientific metadata"):
    from openecon.econometrics.mi.joint import digest

    state = {"schema_version": "mi-pool-v1", "estimates": [[1.0], [1.1]],
             "covariances": [[[0.04]], [[0.05]]], "terms": ["Intercept"], "imputation_ids": ids,
             "complete_df": None, "alpha": 0.05, "imputation_description": "Declared fixed estimands",
             "metadata_json": json.dumps({"precision": "float64", "stata_parity_validated": False,
                 "label": label, "source_results": [{"id": identifier, "data_hash": "f"*64} for identifier in ids]})}
    return {**state, "integrity_sha256": digest(state)}


def test_typed_mi_pool_checks_full_hash_and_structured_identity_metadata(tmp_path):
    first, second = mi_pool([uuid(1), uuid(2)]), mi_pool([uuid(3), uuid(4)])
    report = run(tmp_path, {"one.json": bundle(1), "two.json": bundle(2), "pool.json": first},
                 {"one.json": bundle(3), "two.json": bundle(4), "pool.json": second})
    assert report["validated_mi_states"] == 1
    assert report["exact_scientific_content_equal"]


def test_mi_structured_metadata_scientific_uuid_label_is_not_normalized(tmp_path):
    first, second = mi_pool([uuid(1), uuid(2)], label=uuid(1)), mi_pool([uuid(3), uuid(4)], label=uuid(3))
    with pytest.raises(comparator.ComparisonError, match="metadata_json/\\$json/label"):
        run(tmp_path, {"one.json": bundle(1), "two.json": bundle(2), "pool.json": first},
            {"one.json": bundle(3), "two.json": bundle(4), "pool.json": second})


def test_duplicate_scientific_keys_inside_typed_mi_metadata_are_refused(tmp_path):
    from openecon.econometrics.mi.joint import digest

    first, second = mi_pool([uuid(1), uuid(2)]), mi_pool([uuid(3), uuid(4)])
    first["metadata_json"] = first["metadata_json"].replace('"label": ', '"label": "hidden scientific value", "label": ', 1)
    first["integrity_sha256"] = digest({k: v for k, v in first.items() if k != "integrity_sha256"})
    with pytest.raises(comparator.ComparisonError, match="Duplicate JSON key"):
        run(tmp_path, {"one.json": bundle(1), "two.json": bundle(2), "pool.json": first},
            {"one.json": bundle(3), "two.json": bundle(4), "pool.json": second})


@pytest.mark.parametrize("damage", ["invalid_integrity", "unknown_source_uuid"])
def test_mi_integrity_and_reference_binding_are_required(tmp_path, damage):
    first, second = mi_pool([uuid(1), uuid(2)]), mi_pool([uuid(3), uuid(4)])
    if damage == "invalid_integrity":
        second["estimates"][0][0] += 0.1
    else:
        second = mi_pool([uuid(3), uuid(9)])
    with pytest.raises(comparator.ComparisonError, match="Invalid typed MI state|Unrecognized UUID reference"):
        run(tmp_path, {"one.json": bundle(1), "two.json": bundle(2), "pool.json": first},
            {"one.json": bundle(3), "two.json": bundle(4), "pool.json": second})


@pytest.mark.parametrize("raw,message", [(b'{"value":1,"value":2}', "Duplicate JSON key"),
                                         (b'{"value":NaN}', "Nonfinite JSON constant")])
def test_duplicate_or_nonfinite_json_never_enters_comparison(tmp_path, raw, message):
    with pytest.raises(comparator.ComparisonError, match=message):
        run(tmp_path, {"bad.json": raw}, {"bad.json": raw})


def test_scientific_boolean_and_numeric_values_are_distinct(tmp_path):
    with pytest.raises(comparator.ComparisonError, match="JSON types differ"):
        run(tmp_path, {"science.json": {"value": True}}, {"science.json": {"value": 1}})


@pytest.mark.parametrize("sign", ["", "-"])
def test_distinct_legal_json_exponent_overflows_are_refused_before_equality(tmp_path, sign):
    first = ('{"effect":' + sign + '1e309}').encode()
    second = ('{"effect":' + sign + '2e309}').encode()
    with pytest.raises(comparator.ComparisonError, match="Nonfinite JSON exponent"):
        run(tmp_path, {"scientific.json": first}, {"scientific.json": second})


@pytest.mark.parametrize("sign", ["", "-"])
def test_legal_exponent_overflow_in_integrity_valid_mi_metadata_is_refused(tmp_path, sign):
    from openecon.econometrics.mi.joint import digest

    first, second = mi_pool([uuid(1), uuid(2)]), mi_pool([uuid(3), uuid(4)])
    for state, exponent in [(first, "1e309"), (second, "2e309")]:
        state["metadata_json"] = state["metadata_json"][:-1] + ', "effect":' + sign + exponent + '}'
        state["integrity_sha256"] = digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises(comparator.ComparisonError, match="Nonfinite JSON exponent"):
        run(tmp_path, {"one.json": bundle(1), "two.json": bundle(2), "pool.json": first},
            {"one.json": bundle(3), "two.json": bundle(4), "pool.json": second})
