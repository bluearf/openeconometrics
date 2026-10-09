"""Fisher combination of separately calibrated individual Johansen p-values."""

import math
from numbers import Integral

from openecon.analysis import _coerce_frame, _numeric, _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import chi2_sf
from openecon.resources import plan_workspace


def fisher_johansen(
    *, data, unit, p_value, rank, calibration, statistic="trace", independent=False
):
    """Combine sourced Johansen rank-test p-values for independent panel units.

    Computes -2*sum(log p) with chi-square(2*N) reference. Input p-values must
    test the same rank/deterministic hypothesis and have their own calibration.
    This route does not interpolate sparse critical tables or manufacture
    MacKinnon-Haug-Michelis p-values. Dependence requires a separate joint null.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Supply resident calibrated unit summaries.")
    if independent is not True or statistic not in {"trace", "max_eigenvalue"}:
        raise AnalysisError(
            "invalid_assumptions", "Declare independent=True and trace/max_eigenvalue rank tests."
        )
    if isinstance(rank, bool) or not isinstance(rank, Integral) or not 0 <= rank <= 100:
        raise AnalysisError("invalid_rank", "rank must be an integer in 0..100.")
    if not isinstance(calibration, str) or not calibration.strip():
        raise AnalysisError(
            "missing_calibration",
            "Name the original unit p-value source, version, deterministic and null calibration.",
        )
    frame = _coerce_frame(data)
    if (
        frame.columns.has_duplicates
        or unit == p_value
        or any(c not in frame for c in (unit, p_value))
    ):
        raise AnalysisError(
            "invalid_columns", "Distinct unique unit and p-value columns are required."
        )
    if not 2 <= len(frame) <= 100_000 or frame[unit].isna().any() or frame[unit].duplicated().any():
        raise AnalysisError("invalid_units", "Supply 2..100000 distinct complete unit summaries.")
    plan = plan_workspace(
        "Fisher independent rank-test summaries",
        {
            "pvalue_and_log_buffers": len(frame) * 64,
            "selected_summary_copy": int(
                frame[[unit, p_value]].memory_usage(index=True, deep=True).sum()
            )
            * 2,
        },
    )
    p = _numeric(frame[p_value], p_value)
    if not bool(((p > 0) & (p <= 1)).all()):
        raise AnalysisError(
            "invalid_pvalues",
            "P-values must be finite in (0,1]; zero tails are not silently clipped.",
        )
    value = -2 * float(p.log().sum())
    if not math.isfinite(value):
        raise AnalysisError(
            "invalid_pvalues", "Fisher statistic overflowed; provide resolved p-values."
        )
    return saved_summary(
        TableSet(
            {
                "test": table(
                    [
                        dict(
                            statistic=value,
                            df=2 * len(p),
                            p_value=chi2_sf(value, 2 * len(p)),
                            rank=int(rank),
                        )
                    ]
                ),
                "units": table(frame[[unit, p_value]].to_dict("records")),
            },
            title="Fisher–Johansen combination",
            individual_statistic=statistic,
            unit_calibration=calibration,
            independent_units_assumed=True,
            resource_plan=plan.record(),
            null="Every supplied unit has cointegration rank at most the declared rank",
            pvalue_origin="supplied separately calibrated summaries; no table interpolation",
            sample_hash=_frame_hasher(frame[[unit, p_value]]).hexdigest(),
            source="https://help.eviews.com/content/coint-Panel_Cointegration_Testing.html",
        )
    )
