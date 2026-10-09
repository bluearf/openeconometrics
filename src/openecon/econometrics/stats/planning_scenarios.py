"""Prespecified scenario tables and plots for native prospective designs."""

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.summary_state import saved_summary
from .planning import power_mean
from .power_designs import power_anova, power_regression, power_cluster_mean
from .planning_extended import power_paired_mean, power_logrank

def planning_scenarios(method, scenarios):
    """Tabulate 1..100 explicit prospective designs; no data/model-derived power."""
    from .planning import power_proportion, power_correlation, precision_mean

    routes = {
        "power_mean": power_mean,
        "power_proportion": power_proportion,
        "power_correlation": power_correlation,
        "precision_mean": precision_mean,
        "power_anova": power_anova,
        "power_regression": power_regression,
        "power_paired_mean": power_paired_mean,
        "power_cluster_mean": power_cluster_mean,
        "power_logrank": power_logrank,
    }
    if (
        method not in routes
        or not isinstance(scenarios, list)
        or not 1 <= len(scenarios) <= 100
        or any(not isinstance(s, dict) for s in scenarios)
    ):
        raise AnalysisError(
            "invalid_option", "Use a supported method and 1..100 scalar argument dictionaries."
        )
    rows = []
    settings = []
    for i, design in enumerate(scenarios):
        output = routes[method](**design)
        rows.append({"scenario": i, **output["plan"].iloc[0].to_dict()})
        settings.append(output.attrs)
    return saved_summary(
        TableSet({"scenarios": table(rows)}, method=method, prospective=True, settings=settings)
    )


def planning_plot(output, *, x="n", y="power"):
    """Editable prespecified scenario curve, with no downsampling or observed data."""
    from openecon_charts.charts import line

    if (
        not isinstance(output, TableSet)
        or not output.attrs.get("prospective")
        or "scenarios" not in output
    ):
        raise AnalysisError("invalid_result", "Pass a prospective design/scenario TableSet.")
    data = output["scenarios"]
    if x not in data or y not in data or len(data) < 2:
        raise AnalysisError(
            "invalid_columns", "Select existing numeric scenario axes and at least two rows."
        )
    return line(data=data, x=x, y=y, title="Prospective design assumptions")

