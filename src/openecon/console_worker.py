"""Persistent Python worker. Isolation is lifecycle management, not a sandbox."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import traceback
from uuid import uuid4
from datetime import datetime, timezone

MAX_STDOUT = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_OUTPUTS = 20
MAX_COMMAND_BYTES = 300 * 1024
MAX_WORKER_BYTES = 3 * 1024 * 1024


class WorkerProtocolError(ValueError):
    """Invalid child data; never deserialize executable Python objects."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkerProtocolError("Duplicate JSON field.")
        result[key] = value
    return result


def _reject_constant(value):
    raise WorkerProtocolError("Non-finite JSON number.")


def send_json(connection, value: dict, *, maximum: int = MAX_WORKER_BYTES):
    """The executing process can reach this pipe, so pickle is forbidden."""
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise WorkerProtocolError("The worker message cannot be encoded.") from exc
    if len(payload) > maximum:
        raise WorkerProtocolError("The worker message exceeds its limit.")
    connection.send_bytes(payload)


def receive_json(connection, *, maximum: int = MAX_WORKER_BYTES) -> dict:
    try:
        payload = connection.recv_bytes(maxlength=maximum)
        result = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise WorkerProtocolError("The worker returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise WorkerProtocolError("The worker message must be an object.")
    # Bounded JSON size alone still allows pathological recursive structures.
    pending = [(result, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 250_000 or depth > 64:
            raise WorkerProtocolError("The worker message is too complex.")
        if isinstance(value, dict):
            try:
                for key in value:
                    key.encode("utf-8")
            except UnicodeError as exc:
                raise WorkerProtocolError("Invalid worker text encoding.") from exc
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
        elif isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise WorkerProtocolError("Invalid worker text encoding.") from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise WorkerProtocolError("Non-finite worker value.")
    return result


def _text(value, maximum):
    try:
        return isinstance(value, str) and len(value.encode("utf-8")) <= maximum
    except UnicodeError:
        return False


def validate_ready(message: dict) -> dict:
    if message.get("kind") != "ready" or type(message.get("ready")) is not bool:
        raise WorkerProtocolError("Invalid worker startup message.")
    if message["ready"]:
        group = message.get("process_group")
        if set(message) != {"kind", "ready", "process_group"} or not (
            group is None or (type(group) is int and group > 0)
        ):
            raise WorkerProtocolError("Invalid worker process identity.")
    elif set(message) != {"kind", "ready", "error"} or not _text(message["error"], 8000):
        raise WorkerProtocolError("Invalid worker startup error.")
    return message


def validate_install_request(message: dict, execution_id: str) -> dict:
    from openecon.project_packages import PackageError, _records, canonical_name, canonical_version
    required = {"kind", "id", "request_id", "requirements", "loaded_versions"}
    optional = {"installer", "specifications", "upgrade", "source_options"}
    if (not required <= set(message) or not set(message) <= required | optional
            or message.get("kind") != "package_install" or message.get("id") != execution_id
            or not _text(message.get("request_id"), 100)
            or not isinstance(message.get("loaded_versions"), dict)
            or len(message["loaded_versions"]) > 100):
        raise WorkerProtocolError("Invalid project package request.")
    try:
        requirements = _records(message["requirements"], nullable=True)
        if not requirements:
            raise PackageError("INVALID_PACKAGE", "No packages requested.")
        loaded = {canonical_name(name): canonical_version(value)
                  for name, value in message["loaded_versions"].items()}
        if len(loaded) != len(message["loaded_versions"]):
            raise PackageError("INVALID_PACKAGE", "Duplicate loaded package names.")
        installer = message.get("installer", "pip")
        if "source_options" in message:
            from openecon.package_sources import validate_sources
            message = {**message, "source_options": validate_sources(message["source_options"], {row["name"] for row in requirements})}
        if not isinstance(installer, str) or installer not in {"pip", "uv"} or type(message.get("upgrade", False)) is not bool:
            raise PackageError("INVALID_PACKAGE", "Invalid package installer options.")
        if "specifications" in message:
            from openecon.package_requirements import parse_specifications
            rows, specs = parse_specifications(message["specifications"])
            if _records(rows, nullable=True) != requirements:
                raise PackageError("INVALID_PACKAGE", "Package requirements do not match their specifications.")
            message = {**message, "specifications": specs}
    except (PackageError, TypeError) as exc:
        raise WorkerProtocolError("Invalid project package requirements.") from exc
    return {**message, "requirements": requirements, "loaded_versions": loaded}


def _validate_install_reply(message: dict, execution_id: str, request_id: str) -> dict:
    from openecon.project_packages import PackageError, _records
    if (set(message) != {"kind", "id", "request_id", "ok", "path", "installed", "error"}
            or message.get("kind") != "package_result" or message.get("id") != execution_id
            or message.get("request_id") != request_id or type(message.get("ok")) is not bool):
        raise WorkerProtocolError("Invalid project package response.")
    if message["ok"]:
        if message["error"] is not None or not (message["path"] is None or _text(message["path"], 8000)):
            raise WorkerProtocolError("Invalid project package environment.")
        try:
            _records(message["installed"])
        except PackageError as exc:
            raise WorkerProtocolError("Invalid installed package records.") from exc
    else:
        error = message["error"]
        if (message["path"] is not None or message["installed"] != []
                or not isinstance(error, dict) or set(error) != {"code", "message"}
                or not _text(error["code"], 200) or not _text(error["message"], 4000)):
            raise WorkerProtocolError("Invalid package installation error.")
    return message


def _loaded_overlay_versions(paths: list[str]) -> dict[str, str]:
    """Pin an imported overlay's dependency set to avoid stale Python modules."""
    if not paths:
        return {}
    import importlib.metadata
    from pathlib import Path
    import sys
    from openecon.project_packages import canonical_name, canonical_version
    roots = [Path(path).resolve() for path in paths]
    used = set()
    for module in list(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if isinstance(location, str):
            path = Path(location).resolve()
            used.update(index for index, root in enumerate(roots) if path.is_relative_to(root))
    versions = {}
    for index in used:
        for distribution in importlib.metadata.distributions(path=[str(roots[index])]):
            versions[canonical_name(distribution.metadata["Name"])] = canonical_version(distribution.version)
    return versions


def validate_result(message: dict, execution_id: str) -> dict:
    if (set(message) != {"kind", "id", "result"} or message.get("kind") != "result"
            or message.get("id") != execution_id or not isinstance(message.get("result"), dict)):
        raise WorkerProtocolError("The worker response does not match this execution.")
    result = message["result"]
    allowed = {"status", "stdout", "error", "outputs", "variables", "completed_at", "events"}
    required = allowed - {"completed_at", "events"}
    if (not required <= set(result) or not set(result) <= allowed
            or not isinstance(result["status"], str) or result["status"] not in {"ok", "error"}
            or not _text(result["stdout"], MAX_STDOUT + 100)
            or ("completed_at" in result and not _text(result["completed_at"], 100))):
        raise WorkerProtocolError("Invalid worker result fields.")
    error = result["error"]
    if error is not None and (
        not isinstance(error, dict) or set(error) != {"type", "message", "traceback"}
        or not _text(error.get("type"), 2000) or not _text(error.get("message"), 16000)
        or not _text(error.get("traceback"), 64000)
    ):
        raise WorkerProtocolError("Invalid worker error.")
    if (result["status"] == "ok") != (error is None):
        raise WorkerProtocolError("Inconsistent worker status.")
    outputs, variables = result["outputs"], result["variables"]
    if not isinstance(outputs, list) or len(outputs) > MAX_OUTPUTS:
        raise WorkerProtocolError("Invalid worker outputs.")
    if len(json.dumps(outputs, ensure_ascii=False).encode("utf-8")) > MAX_OUTPUT_BYTES + 100:
        raise WorkerProtocolError("Oversized worker outputs.")
    from openecon.output_latex import validate_latex_fields
    for output in outputs:
        if (not isinstance(output, dict) or not {"type", "data"} <= set(output)
                or not set(output) <= {"type", "data", "latex", "latex_math", "latex_style", "latex_notes"}
                or not isinstance(output["type"], str)
                or output["type"] not in {"table", "model", "plot", "text", "latex"}):
            raise WorkerProtocolError("Invalid worker display.")
        if output["type"] in {"table", "model", "plot"} and not isinstance(output["data"], dict):
            raise WorkerProtocolError("Invalid structured worker display.")
        if output["type"] in {"text", "latex"} and not isinstance(output["data"], str):
            raise WorkerProtocolError("Invalid text worker display.")
        if output["type"] == "plot" and "artifact" in output["data"]:
            from openecon.plot_artifacts import validate_reference
            try:
                validate_reference(output["data"]["artifact"])
                if output["data"].get("kind") != "network":
                    raise ValueError("Only network plots can reference a local artifact.")
            except ValueError as exc:
                raise WorkerProtocolError("Invalid local network plot reference.") from exc
        try:
            validate_latex_fields(output)
        except ValueError as exc:
            raise WorkerProtocolError("Invalid worker LaTeX.") from exc
    if not isinstance(variables, list) or len(variables) > 100:
        raise WorkerProtocolError("Invalid worker variables.")
    if "events" in result:
        from openecon.output_events import validate_output_events
        try:
            validate_output_events(result["events"], result["stdout"], outputs)
        except ValueError as exc:
            raise WorkerProtocolError("Invalid worker output timeline.") from exc
    for variable in variables:
        if (not isinstance(variable, dict) or set(variable) != {"name", "type", "preview"}
                or not all(_text(variable[key], 1000) for key in variable)):
            raise WorkerProtocolError("Invalid worker variable.")
    return result


def validate_command(message: dict) -> dict:
    if message == {"op": "close"}:
        return message
    if (set(message) != {"op", "code", "id"} or message["op"] != "execute"
            or not isinstance(message["code"], str) or not 1 <= len(message["code"]) <= 64000
            or not _text(message["id"], 100)):
        raise WorkerProtocolError("Invalid execution command.")
    return message


class BoundedText(io.TextIOBase):
    def __init__(self):
        self.parts = []
        self.length = 0
        self.truncated = False

    def write(self, text):
        text = str(text)
        if self.truncated:
            return len(text)
        remaining = MAX_STDOUT - self.length
        if remaining > 0:
            encoded = text.encode("utf-8", errors="replace")
            part = encoded[:remaining].decode("utf-8", errors="ignore")
            self.parts.append(part)
            self.length += len(part.encode("utf-8"))
        if len(text.encode("utf-8", errors="replace")) > remaining:
            self.truncated = True
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts) + ("\n[Output truncated at 64 KiB.]" if self.truncated else "")


def _safe_scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value[:500] if isinstance(value, str) else value
    if isinstance(value, int):
        # JSON numbers outside this range lose integer precision in the UI.
        return str(value) if abs(value) > 2**53 - 1 else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if hasattr(value, "item"):
            return _safe_scalar(value.item())
        return str(value)[:500]
    except Exception:
        return f"<{type(value).__name__}>"


def _variables(namespace):
    import pandas as pd
    import torch
    from openecon.dataset import Dataset
    from openecon.models import ResultBundle
    items = []
    hidden = {"oe", "pd", "torch", "display", "In", "Out"}
    for name, value in list(namespace.items()):
        if name.startswith("_") or name in hidden:
            continue
        if isinstance(value, Dataset):
            rows = f"{value.row_count:,}" if value.row_count is not None else "unknown"
            preview = f"{rows} rows × {len(value.columns)} columns · batch source"
        elif isinstance(value, pd.DataFrame):
            preview = f"{len(value):,} rows × {len(value.columns)} columns"
        elif isinstance(value, pd.Series):
            preview = f"{len(value):,} values · {value.dtype}"
        elif isinstance(value, torch.Tensor):
            preview = f"shape={tuple(value.shape)} · {value.dtype}"
        elif isinstance(value, ResultBundle):
            preview = f"{value.spec.estimator.upper()} · {value.nobs:,} observations"
        elif isinstance(value, (str, int, float, bool, type(None))):
            try:
                preview = repr(value)[:160]
            except Exception:
                preview = f"<{type(value).__name__}>"
        elif isinstance(value, (list, tuple, dict, set)):
            preview = f"{len(value):,} items"
        else:
            preview = f"<{type(value).__name__}>"
        items.append({"name": name[:200], "type": type(value).__name__[:200], "preview": preview[:200]})
        if len(items) >= 100:
            break
    return items


def _execute(code: str, namespace: dict, execution_id: str, *, artifact_workspace=None) -> dict:
    import pandas as pd
    from openecon.dataset import Dataset
    from openecon.models import ResultBundle
    from openecon.plotting import PlotSpec
    from openecon.networks import Network
    from openecon._network_flow import NetworkFlowResult
    from openecon._network_cost_flow import CostFlowResult
    from openecon._network_cut import NetworkCutResult
    from openecon._network_sbm import NetworkBlockResult
    from openecon._network_model_common import ModelResult
    from openecon._network_temporal import NetworkSnapshots
    from openecon.latex import Latex
    from openecon.econometrics.bayesian.posterior import PosteriorBundle, PosteriorContrast, PosteriorDraws
    from openecon.output_latex import MAX_MATH_BYTES, add_output_latex, validate_latex_fields
    output = BoundedText()
    displays = []
    events = []
    stdout_position = 0
    output_bytes = 0

    def flush_stdout():
        nonlocal stdout_position
        text = output.getvalue()
        if len(text) > stdout_position:
            events.append({"type": "stdout", "text": text[stdout_position:]})
            stdout_position = len(text)

    def display(value):
        nonlocal output_bytes
        if value is None:
            return
        if len(displays) >= MAX_OUTPUTS:
            output.write("\n[Display limit reached: 20 outputs.]\n")
            return
        if isinstance(value, (Network, NetworkFlowResult, CostFlowResult, NetworkCutResult, NetworkBlockResult, NetworkSnapshots, ModelResult)):
            return display(value.summary())
        if isinstance(value, (PosteriorBundle, PosteriorContrast, PosteriorDraws)):
            return display(value.summary())
        if isinstance(value, Latex):
            math = getattr(value, "math", None)
            if isinstance(math, str) and len(math.encode("utf-8")) > MAX_MATH_BYTES:
                math = None
            item = {"type": "latex", "data": str(value), "latex": str(value),
                    "latex_math": math}
            if getattr(value, "notes", None):
                item["latex_notes"] = value.notes
        elif isinstance(value, PlotSpec):
            item = {"type": "plot", "data": value.transport_dump()}
        elif isinstance(value, ResultBundle):
            payload = value.model_dump(mode="json", exclude={"sample_positions", "covariance_matrix"})
            payload["display_omitted"] = ["sample_positions", "covariance_matrix"]
            item = {"type": "model", "data": payload}
        elif isinstance(value, (pd.DataFrame, pd.Series, Dataset)):
            streamed = isinstance(value, Dataset)
            frame = value.head(50) if streamed else value.to_frame() if isinstance(value, pd.Series) else value
            view = frame.iloc[:50, :30]
            rows = ([[_safe_scalar(cell) for cell in row]
                     for row in view.itertuples(index=False, name=None)]
                    if len(view.columns) else [[] for _ in range(len(view))])
            item = {"type": "table", "data": {"columns": [str(column) for column in view.columns],
                    "rows": rows,
                    "index_names": [str(name) if name is not None else None for name in view.index.names],
                    "index": [[_safe_scalar(level) for level in label]
                              if isinstance(view.index, pd.MultiIndex) else [_safe_scalar(label)]
                              for label in view.index],
                    "total_rows": value.row_count if streamed and value.row_count is not None else len(frame),
                    "total_columns": len(value.columns) if streamed else len(frame.columns)}}
            if streamed and value.row_count is None:
                item["data"]["total_rows_known"] = False
            if "publication_notes" in frame.attrs:
                item["data"]["publication_notes"] = frame.attrs["publication_notes"]
        else:
            item = {"type": "text", "data": str(value)[:16000]}
        try:
            add_output_latex(item)
            validate_latex_fields(item)
        except ValueError:
            output.write("\n[LaTeX display limit reached; export the full result to a file.]\n")
            return
        serialized = json.dumps(item, ensure_ascii=False, allow_nan=False)
        if (artifact_workspace is not None and isinstance(value, PlotSpec)
                and value.kind == "network"):
            from openecon.plot_artifacts import INLINE_PLOT_BYTES, store_plot
            if len(serialized.encode("utf-8")) > INLINE_PLOT_BYTES:
                item["data"] = store_plot(artifact_workspace, item["data"])
                serialized = json.dumps(item, ensure_ascii=False, allow_nan=False)
        if output_bytes + len(serialized.encode("utf-8")) > MAX_OUTPUT_BYTES:
            output.write("\n[Display limit reached: 2 MiB.]\n")
            return
        output_bytes += len(serialized.encode("utf-8"))
        flush_stdout()
        events.append({"type": "output", "index": len(displays)})
        displays.append(item)

    namespace["display"] = display
    error = None
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            from openecon.script_packages import rewrite_install_commands
            tree = ast.parse(rewrite_install_commands(code), filename=f"<openecon:{execution_id}>", mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expression = tree.body.pop()
                exec(compile(tree, f"<openecon:{execution_id}>", "exec"), namespace)
                value = eval(compile(ast.Expression(expression.value), f"<openecon:{execution_id}>", "eval"), namespace)
                namespace["_"] = value
                display(value)
            else:
                exec(compile(tree, f"<openecon:{execution_id}>", "exec"), namespace)
        except BaseException as exc:
            error = {"type": type(exc).__name__[:500], "message": str(exc)[:4000],
                     "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-16000:]}
    flush_stdout()
    return {"status": "error" if error else "ok", "stdout": output.getvalue(), "error": error,
            "outputs": displays, "events": events, "variables": _variables(namespace),
            "completed_at": datetime.now(timezone.utc).isoformat()}


def worker_main(connection, workspace: str, environment: dict[str, str] | None = None,
                package_path: str | None = None, scratch_directory: str | None = None):
    # Native writes/subprocess inheritance cannot flood the server's stdout.
    # Python print/logging output is captured per execution below.
    try:
        if environment is not None:
            os.environ.clear()
            os.environ.update(environment)
        if os.name == "posix":
            os.setsid()
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        os.environ["OPENECON_WORKSPACE"] = workspace
        if scratch_directory is not None:
            os.environ["OPENECON_SCRATCH_DIRECTORY"] = scratch_directory
        else:
            os.environ.pop("OPENECON_SCRATCH_DIRECTORY", None)
        os.chdir(workspace)
        if package_path is not None:
            import importlib
            import sys
            from pathlib import Path
            from openecon.project_packages import validate_overlay
            validate_overlay(Path(package_path))
            sys.path.insert(0, package_path)
            importlib.invalidate_caches()
        import openecon as oe
        import pandas as pd
        import torch
        torch.set_num_threads(1)
        namespace = {"__name__": "__main__", "__builtins__": __builtins__, "oe": oe, "pd": pd, "torch": torch}
        send_json(connection, {"kind": "ready", "ready": True,
                              "process_group": os.getpgrp() if os.name == "posix" else None})
    except BaseException as exc:
        send_json(connection, {"kind": "ready", "ready": False,
                              "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
        connection.close()
        return
    try:
        package_paths = [package_path] if package_path is not None else []

        def install_for_script(requirements, execution_id, **options):
            import importlib
            from pathlib import Path
            import sys
            from openecon.project_packages import PackageError, validate_overlay
            request_id = uuid4().hex
            send_json(connection, {"kind": "package_install", "id": execution_id,
                                   "request_id": request_id, "requirements": requirements,
                                   "loaded_versions": _loaded_overlay_versions(package_paths), **options})
            reply = _validate_install_reply(receive_json(connection, maximum=MAX_COMMAND_BYTES),
                                            execution_id, request_id)
            if not reply["ok"]:
                raise PackageError(reply["error"]["code"], reply["error"]["message"])
            overlay = reply["path"]
            if overlay is not None:
                validate_overlay(Path(overlay))
                # Imported unchanged modules keep their original __path__.
                # New imports resolve from the current generation first.
                sys.path[:] = [value for value in sys.path if value not in package_paths]
                sys.path.insert(0, overlay)
                if overlay not in package_paths:
                    package_paths.append(overlay)
                importlib.invalidate_caches()
            return {"installed": reply["installed"]}

        while True:
            command = validate_command(receive_json(connection, maximum=MAX_COMMAND_BYTES))
            if command.get("op") == "close":
                break
            if command.get("op") == "execute":
                try:
                    from openecon.script_packages import _use_installer
                    with _use_installer(lambda requirements, **options: install_for_script(requirements, command["id"], **options)):
                        result = _execute(command["code"], namespace, command["id"],
                                          artifact_workspace=workspace)
                    send_json(connection, {"kind": "result", "id": command["id"], "result": result})
                except BaseException as exc:
                    send_json(connection, {"kind": "result", "id": command["id"], "result": {
                        "status": "error", "stdout": "", "outputs": [], "variables": [],
                        "error": {"type": type(exc).__name__[:500], "message": str(exc)[:4000], "traceback": ""}}})
    except (EOFError, BrokenPipeError, OSError, WorkerProtocolError):
        pass
    finally:
        connection.close()
