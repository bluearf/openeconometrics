"""Original physical CART reference inputs. Stdlib loader; no fit/RNG on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET_SHA256 = "a41fce71332b4a3937f1fcb335cfeb4f6de3256db5ecd8988a99bc35dde7cb36"
REFERENCE_SHA256 = "8d9f5904ffeb55f51baf99bbc38860b038a068ec88ca3b208a90ec030259befd"
ORIGINAL_RAW_SHA256 = {
    "3.11": "d36c9807549617e85d5319fd4f54e5c793aab726ff9163faee0053b8bebc236e",
    "3.13": "38fb4eeb6617a2400884ac83291ba332ae9e7f46a41b65459ccc6173240e3438",
}
INPUT_FIELDS = ["schema", "n", "features", "outcome", "weight_name", "settings", "split", "source", "classes"]
PROVENANCE_SHA256 = 'd7eb219422cfb8642878ca2723167f4d3484573084e3aa325ba60433417d6e04'
ORIGINAL_FITTED_DIGESTS_BY_CASE = {'regression-8813': {'3.11': '6ed1e7fd019bd90519b097cd729d52b6daa0ac8d0aed5a310851a1595a4e4cd2', '3.13': '761132206be1e43308ad38e6c07977a5a93196d5014a45518370902d31216d5f'}, 'regression-19423': {'3.11': '52538f98659d20362101cef05eaad5b1e4ca888cc92c244d055dff82bc597527', '3.13': 'bd6ff1982ce6b838e0364b88f6cd81e02bce5a7056cb53c7a31e65530b5c70db'}, 'classification-8813': {'3.11': 'd63bea354e659c300481b497031be19e8e80f5652f12ce23e13da6adf05f6366', '3.13': 'd63bea354e659c300481b497031be19e8e80f5652f12ce23e13da6adf05f6366'}, 'classification-19423': {'3.11': '75b2e3ba649a168528af414d25a7e94ea3676113de5239da14911e43da746805', '3.13': '75b2e3ba649a168528af414d25a7e94ea3676113de5239da14911e43da746805'}}
SCHEMA = "openecon.supervised.cart.original-native-inputs.v1"
MAX_PACKET_BYTES = 1_000_000
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Original CART input has a duplicate JSON key.")
        result[key] = value
    return result


def _read_bounded(path):
    path = Path(path)
    if path.stat().st_size > MAX_PACKET_BYTES:
        raise ValueError("Original CART input packet exceeds its fixed byte bound.")
    body = path.read_bytes()
    if len(body) > MAX_PACKET_BYTES:
        raise ValueError("Original CART input packet exceeds its fixed byte bound.")
    return body


def _json(body):
    return json.loads(body, object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite original CART input.")))


def _checked_packet(body, *, _expected_packet_sha256=PACKET_SHA256):
    """Private override exists only for resealed negative controls, never loader."""
    if len(body) > MAX_PACKET_BYTES or hashlib.sha256(body).hexdigest() != _expected_packet_sha256:
        raise ValueError("Original complete CART input packet hash disagrees.")
    packet = _json(body)
    reference_body = _read_bounded(FIXTURES / "supervised-cart-native-reference.json")
    if hashlib.sha256(reference_body).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The complete original native CART reference was altered.")
    reference = _json(reference_body)
    if (set(packet) != {"schema", "provenance", "source_identity_fields", "cases"}
            or packet["schema"] != SCHEMA or packet["source_identity_fields"] != INPUT_FIELDS
            or reference["source_identity_fields"] != INPUT_FIELDS):
        raise ValueError("Original CART packet schema/identity fields disagree.")
    provenance = packet["provenance"]
    if hashlib.sha256(_canonical(provenance)).hexdigest() != PROVENANCE_SHA256:
        raise ValueError("Complete original CART origin metadata was altered.")
    if (provenance["original_git_head"] != "cdf620a5dea7da61df34a70fa1525ac7dabb4094"
            or provenance["authentic_failed_run"] != "37976988484"
            or provenance["authentic_tested_merge"] != "872bbab66f59b155c381db44338095d124ff3325"
            or provenance["original_git_frozen_reference_file_sha256"] != REFERENCE_SHA256
            or provenance["original_full_raw_state_artifact_sha256_by_minor"] != ORIGINAL_RAW_SHA256):
        raise ValueError("Original CART raw/native/Git provenance disagrees.")
    options = reference["metadata"]["synthetic_cases"]
    if len(packet["cases"]) != len(options) or len(options) != len(reference["cases"]) or len(options) != 4:
        raise ValueError("All four complete original CART cases are required.")
    seen = set()
    for case, option, native in zip(packet["cases"], options, reference["cases"], strict=True):
        if set(case) != {"case_id", "options", "complete_physical_identity_by_original_minor", "source_input_sha256", "original_full_fitted_state_digest_by_minor"}:
            raise ValueError("Original CART case fields cannot be omitted or added.")
        case_id = option["task"] + "-" + str(option["seed"])
        if (case["case_id"] != case_id or case_id in seen
                or _canonical(case["options"]) != _canonical(option)
                or case["source_input_sha256"] != native["source_input_sha256"]):
            raise ValueError("Original CART case order/settings/native input binding disagrees.")
        seen.add(case_id)
        originals = case["complete_physical_identity_by_original_minor"]
        if set(originals) != {"3.11", "3.13"}:
            raise ValueError("Both complete original minor input packets are required.")
        for identity in originals.values():
            if (set(identity) != set(INPUT_FIELDS)
                    or hashlib.sha256(_canonical(identity)).hexdigest() != native["source_input_sha256"]):
                raise ValueError("A complete original CART physical source field was altered.")
        if _canonical(originals["3.11"]) != _canonical(originals["3.13"]):
            raise ValueError("Original physical inputs disagree between minors.")
        if case["original_full_fitted_state_digest_by_minor"] != ORIGINAL_FITTED_DIGESTS_BY_CASE[case_id]:
            raise ValueError("Original complete fitted-state provenance was altered.")
        if case["original_full_fitted_state_digest_by_minor"]["3.13"] != native["source_digest"]:
            raise ValueError("Original full fitted state/native author binding disagrees.")
    return packet


def load_original_case(task, seed, *, path=None):
    """Load exact original source/dtypes/index/split/settings/class coordinates."""
    packet = _checked_packet(_read_bounded(path or FIXTURES / "supervised-cart-native-source-inputs.json"))
    if type(task) is not str or type(seed) is not int:
        raise ValueError("Original CART case requires its exact task and integer seed label.")
    for case in packet["cases"]:
        if case["options"]["task"] == task and case["options"]["seed"] == seed:
            # Both complete minor identities were checked byte canonically equal.
            # This returns physical inputs, never a fitted tree/reference cache.
            return case["complete_physical_identity_by_original_minor"]["3.13"]
    raise ValueError("The original CART reference declares only its four fixed cases.")
