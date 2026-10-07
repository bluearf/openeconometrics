"""JSON command-line access to the same analysis engine used by the web app."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

from openecon.optional_dependencies import require_extra
try:
    import typer
except ModuleNotFoundError:
    require_extra("cli", "typer")
    raise
from pydantic import ValidationError

from openecon.analysis import AnalysisError, fit as fit_model
from openecon.data import profile_dataset, profile_frame, read
from openecon.models import ModelSpec

app = typer.Typer(
    name="openecon",
    help="Local, reproducible Python statistics. This initial release is not full Stata parity.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _emit(value: Any) -> None:
    typer.echo(_json(value))


def _fail(error: Exception) -> None:
    """Keep expected user errors machine-readable and tracebacks off stdout."""
    typer.echo(_json({"error": str(error), "type": type(error).__name__}), err=True)
    raise typer.Exit(code=1) from error


@app.command("inspect")
def inspect_file(
    path: Annotated[Path, typer.Argument(help="CSV, Parquet, Excel or Stata file to inspect.")],
) -> None:
    """Print a local file's profile as JSON; this does not import it into a workspace."""
    try:
        frame = read(path)
        from openecon.dataset import Dataset
        profiler = profile_dataset if isinstance(frame, Dataset) else profile_frame
        _emit(profiler(frame, name=path.name, source="upload"))
    except (AnalysisError, ValidationError, ValueError, OSError) as error:
        _fail(error)


@app.command("import")
def import_file(
    path: Annotated[Path, typer.Argument(help="Local file explicitly selected for import.")],
    workspace: Annotated[
        Path, typer.Option(help="Workspace shared by the UI and MCP server.")
    ] = Path(".openecon"),
) -> None:
    """Import a dataset so agents can access it by dataset ID without reading arbitrary paths."""
    from openecon.workspace import Workspace

    try:
        _emit(Workspace(workspace).import_file(path))
    except (AnalysisError, ValidationError, ValueError, OSError) as error:
        _fail(error)


@app.command("fit")
def fit_file(
    path: Annotated[Path, typer.Argument(help="Dataset file to analyze.")],
    outcome: Annotated[str, typer.Option("--outcome", "-y", help="Outcome column.")],
    predictor: Annotated[
        list[str], typer.Option("--predictor", "-x", help="Predictor column; repeat this option.")
    ],
    estimator: Annotated[str, typer.Option(help="Estimator: ols, logit or probit.")] = "ols",
    covariance: Annotated[
        str | None,
        typer.Option(
            help="nonrobust, HC1, HC3 or cluster; default HC3 for OLS, nonrobust for binary, cluster with --cluster."
        ),
    ] = None,
    cluster: Annotated[
        str | None,
        typer.Option(
            help="Cluster column; selects cluster covariance unless explicitly specified."
        ),
    ] = None,
    categorical: Annotated[
        list[str] | None,
        typer.Option("--categorical", help="Categorical predictor; repeat this option."),
    ] = None,
    missing: Annotated[str, typer.Option(help="Missing-data policy: raise or drop.")] = "raise",
    intercept: Annotated[bool, typer.Option("--intercept/--no-intercept")] = True,
    alpha: Annotated[
        float, typer.Option(help="Significance level for confidence intervals.")
    ] = 0.05,
    output: Annotated[
        Path | None, typer.Option(help="Also save the complete result JSON to this file.")
    ] = None,
) -> None:
    """Fit one validated model and print its complete ResultBundle as JSON."""
    try:
        default_covariance = "HC3" if estimator == "ols" else "nonrobust"
        selected_covariance = covariance or ("cluster" if cluster else default_covariance)
        if cluster is not None and selected_covariance != "cluster":
            raise ValueError(
                "--cluster requires --covariance cluster; omit --covariance to select it automatically."
            )
        spec = ModelSpec(
            estimator=estimator,
            outcome=outcome,
            predictors=predictor,
            categorical=categorical or [],
            intercept=intercept,
            covariance=selected_covariance,
            cluster=cluster,
            missing=missing,
            alpha=alpha,
        )
        result = fit_model(spec, data=read(path)).model_dump(mode="json")
        serialized = _json(result)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n", encoding="utf-8")
        typer.echo(serialized)
    except (AnalysisError, ValidationError, ValueError, OSError) as error:
        _fail(error)


@app.command("serve")
def serve(
    host: Annotated[
        str, typer.Option(help="Bind address; local loopback by default.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    workspace: Annotated[Path, typer.Option(help="Local data and result workspace.")] = Path(
        ".openecon"
    ),
) -> None:
    """Start the local web application. Server logs go to stderr."""
    from openecon.optional_dependencies import require_extra
    require_extra("server", "fastapi", "uvicorn", "multipart")
    import uvicorn

    from openecon.server import create_app

    try:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "This alpha only supports loopback hosts: 127.0.0.1, localhost or ::1. "
                "Remote serving requires an authentication and access-control design."
            )
        log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
        log_config["handlers"]["access"]["stream"] = "ext://sys.stderr"
        uvicorn.run(
            create_app(workspace), host=host, port=port, log_level="info", log_config=log_config
        )
    except (ValueError, OSError) as error:
        _fail(error)


@app.command("mcp")
def mcp(
    workspace: Annotated[
        Path, typer.Option(help="Workspace accessible to this MCP client.")
    ] = Path(".openecon"),
) -> None:
    """Run the official MCP SDK STDIO server. Stdout is reserved for protocol messages."""
    from openecon.mcp_server import create_mcp_server

    try:
        create_mcp_server(workspace).run(transport="stdio")
    except (ValueError, OSError) as error:
        _fail(error)


if __name__ == "__main__":
    app()
