"""Bounded, aggregate-only publication exports from saved MCP results."""

from __future__ import annotations

import json
import math

from openecon.data import DataError
from openecon.models import ResultBundle
from openecon.output_latex import MAX_LATEX_BYTES, MAX_MATH_BYTES

MAX_LATEX_INPUT_BYTES = 256 * 1024
MAX_LATEX_COEFFICIENTS = 256

_MODEL_FIELDS = (
    "id",
    "created_at",
    "spec",
    "nobs",
    "nobs_original",
    "dropped_rows",
    "coefficients",
    "metrics",
    "warnings",
    "title",
)
_COEFFICIENT_FIELDS = (
    "term",
    "estimate",
    "std_error",
    "statistic",
    "p_value",
    "ci_low",
    "ci_high",
    "equation",
)
_INFERENCE_SCALARS = (
    "available",
    "target",
    "covariance",
    "use_t",
    "df_inference",
    "alpha",
    "confidence_level",
    "cluster_column",
    "cluster_count",
    "correction",
    "small_sample_correction",
    "df_resid",
    "kernel",
    "lags",
    "bandwidth",
    "lag_selection",
    "time_gaps",
    "periods",
    "effective_covariance",
    "absorbed_degrees_of_freedom",
    "degrees_of_freedom_convention",
    "sigma",
    "reference",
    "r_squared_definition",
    "scheme",
    "reps",
    "successful_reps",
    "failed_reps",
    "seed",
    "frequency_count_resampling",
    "hansen",
)
_EXTRA_SCALARS = (
    "target",
    "model",
    "absorbed_effects",
    "absorbed_degrees_of_freedom",
    "family",
    "link",
    "distribution",
    "metric",
    "method",
    "likelihood",
    "partial_likelihood",
    "ties",
    "log_likelihood_scale",
    "parameterization",
    "alpha_test_note",
    "vuong_note",
    "excluded_lag_window_rows",
    "missing_rows",
)


def _unavailable():
    return DataError(
        "This saved result cannot safely produce a coefficient LaTeX table with the current "
        "renderer. Retrieve it without include_latex; no refit or saved record was changed.",
        "LATEX_UNAVAILABLE",
    )


def _limit():
    return DataError(
        "The complete saved-model LaTeX export exceeds the MCP presentation limit. "
        "Retrieve the compact result without include_latex and export the full table locally "
        "with ResultBundle.to_latex(). No table was truncated.",
        "LATEX_LIMIT",
    )


def _scalars(mapping, fields):
    if not isinstance(mapping, dict):
        raise _unavailable()
    projected = {}
    for key in fields:
        if key in mapping:
            value = mapping[key]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise _unavailable()
            projected[key] = value
    return projected


def _list(mapping, key, allowed):
    if key not in mapping:
        return {}
    value = mapping[key]
    if not isinstance(value, list) or not all(isinstance(item, allowed) for item in value):
        raise _unavailable()
    return {key: value}


def result_latex_fields(result: dict) -> dict:
    """Regenerate escaped TeX from aggregate values, never saved TeX or prediction state.

    The source and mathematical view are complete within explicit byte limits.
    Missing OLS distribution and unavailable/prediction-only coefficient
    inference are rejected because the publication renderer would otherwise
    describe assumed inference or expose hidden smoother query rows.
    """
    try:
        info, extra = result.get("inference", {}), result.get("extra", {})
        if not isinstance(info, dict) or not isinstance(extra, dict):
            raise _unavailable()
        if extra.get("target") == "prediction" or info.get("target") == "prediction":
            raise DataError(
                "Prediction-only LaTeX tables are not available through MCP because they may "
                "contain observation or query rows. Retrieve the aggregate result without include_latex.",
                "LATEX_UNAVAILABLE",
            )
        if info.get("available") is False:
            raise DataError(
                "This result records unavailable coefficient inference. MCP LaTeX coefficient "
                "tables require recorded inference; retrieve the result without include_latex.",
                "LATEX_UNAVAILABLE",
            )
        spec = result.get("spec", {})
        if not isinstance(spec, dict) or not (info.get("covariance") or spec.get("covariance")):
            raise _unavailable()
        if spec.get("estimator", "ols") == "ols" and type(info.get("use_t")) is not bool:
            raise DataError(
                "The saved OLS inference distribution is unavailable. Retrieve the result "
                "without include_latex instead of assuming normal or Student t inference.",
                "LATEX_UNAVAILABLE",
            )
        coefficients = result.get("coefficients")
        if not isinstance(coefficients, list):
            raise _unavailable()
        if len(coefficients) > MAX_LATEX_COEFFICIENTS:
            raise _limit()
        fields = {key: result[key] for key in _MODEL_FIELDS if key in result}
        fields["coefficients"] = [_scalars(item, _COEFFICIENT_FIELDS) for item in coefficients]
        fields["predictions"] = fields["sample_positions"] = fields["covariance_matrix"] = []
        fields["inference"] = _scalars(info, _INFERENCE_SCALARS)
        for key, allowed in (
            ("notes", str),
            ("cluster_columns", str),
            ("cluster_counts", (int, float)),
            ("coefficient_df", (int, float)),
            ("coefficient_scale", (int, float)),
        ):
            fields["inference"].update(_list(info, key, allowed))
        fields["extra"] = _scalars(extra, _EXTRA_SCALARS)
        fields["extra"].update(_list(extra, "notes", str))
        provenance = result.get("provenance", {})
        fields["provenance"] = _scalars(provenance, ("sample_position_count",))
        fields["tests"] = {}
        test = result.get("tests", {}).get("model")
        if test and test.get("distribution") in {"F", "chi2"}:
            required = ("distribution", "statistic", "df", "p_value") + (
                ("df2",) if test["distribution"] == "F" else ()
            )
            if any(key not in test or test[key] is None for key in required):
                raise _unavailable()
            numeric = {key: test[key] for key in required if key != "distribution"}
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numeric.values()
            ):
                raise _unavailable()
            if (
                numeric["df"] <= 0
                or numeric.get("df2", 1) <= 0
                or numeric["statistic"] < 0
                or not 0 <= numeric["p_value"] <= 1
            ):
                raise _unavailable()
            fields["tests"]["model"] = _scalars(test, required)
        # Plain JSON removes raw Latex subclasses from ordinary labels. Only
        # renderer-generated commands enter the export; no buf/file is used.
        encoded = json.dumps(fields, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_LATEX_INPUT_BYTES:
            raise _limit()
        model = ResultBundle.model_validate(json.loads(encoded))
        from openecon.latex import display_latex

        source, latex_math = display_latex(model, max_rows=None, max_columns=None)
        if provenance.get("synthetic_data") is True or provenance.get("source") == "example":
            source = model.to_latex(
                notes=[
                    "Synthetic example data; this table does not describe observed research data."
                ]
            )
        notes = source.notes
        if (
            len(str(source).encode("utf-8")) > MAX_LATEX_BYTES
            or not isinstance(latex_math, str)
            or len(latex_math.encode("utf-8")) > MAX_MATH_BYTES
            or len(json.dumps(notes, ensure_ascii=False).encode("utf-8")) > MAX_MATH_BYTES
        ):
            raise _limit()
        packages = ["amsmath", "booktabs"]
        for environment, package in (("adjustbox", "adjustbox"), ("longtable", "longtable")):
            if "\\begin{" + environment + "}" in source:
                packages.append(package)
        if r"\arraybackslash" in source:
            packages.append("array")
        return {
            "latex": str(source),
            "latex_math": latex_math,
            "latex_notes": notes,
            "latex_packages": packages,
        }
    except DataError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
        raise _unavailable() from error
