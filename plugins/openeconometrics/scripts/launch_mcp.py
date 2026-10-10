#!/usr/bin/env python3
"""Start an explicitly configured OpenEconometrics MCP runtime without installing it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

MAX_CONFIG_BYTES = 64 * 1024


class ConfigurationError(ValueError):
    """A configuration cannot select a supported local runtime and workspace."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError("Duplicate configuration keys are not allowed.")
        result[key] = value
    return result


def _finite_json(value: str) -> None:
    raise ConfigurationError("Nonfinite JSON values are not allowed.")


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ConfigurationError(f"The {label} must be an absolute path.")
    path = Path(value)
    if not path.is_absolute():
        raise ConfigurationError(f"The {label} must be an absolute path.")
    return path


def configured_launch(path: Path) -> list[str]:
    """Validate a bounded, strict copy of the workbench's mcp_server entry."""
    _absolute_path(str(path), "configuration file")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
    except OSError:
        raise ConfigurationError(
            "The plugin configuration is missing or unreadable. Open Connect agent in "
            "OpenEconometrics and configure its runtime and workspace before starting."
        ) from None
    if len(raw) > MAX_CONFIG_BYTES:
        raise ConfigurationError("The plugin configuration exceeds 64 KiB.")
    try:
        config = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_finite_json
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ConfigurationError("The plugin configuration must contain valid UTF-8 JSON.") from None
    if not isinstance(config, dict) or set(config) - {"schema_version", "command", "args"}:
        raise ConfigurationError("Only schema_version, command and args are supported.")
    if "schema_version" in config and (
        type(config["schema_version"]) is not int or config["schema_version"] != 1
    ):
        raise ConfigurationError("The plugin configuration schema_version must be 1.")
    command = _absolute_path(config.get("command"), "runtime executable")
    if (
        not command.is_file()
        or not os.access(command, os.X_OK)
        or (os.name == "nt" and command.suffix.lower() not in {".exe", ".com"})
    ):
        raise ConfigurationError("The configured runtime is missing or is not executable.")
    args = config.get("args")
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise ConfigurationError("The configured runtime args must be a list of strings.")
    frozen = len(args) == 3 and args[:2] == ["mcp", "--workspace"]
    source = (
        len(args) == 5
        and args[0] == "-m"
        and args[1] in {"openecon.cli", "openecon.desktop_entry"}
        and args[2:4] == ["mcp", "--workspace"]
    )
    if not (frozen or source):
        raise ConfigurationError(
            "Use the OpenEconometrics mcp --workspace launch arguments from Connect agent."
        )
    workspace = _absolute_path(args[-1], "workspace")
    if not workspace.is_dir():
        raise ConfigurationError("The configured workspace must be an existing directory.")
    return [str(command), *args]


def _run_windows(launch: list[str]) -> int:
    """Keep inherited protocol pipes and stop the child if the launcher is interrupted."""
    child = subprocess.Popen(launch)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    try:
        try:
            return child.wait()
        except KeyboardInterrupt:
            return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Absolute path to the plugin runtime configuration.")
    arguments = parser.parse_args(argv)
    selected = (
        arguments.config
        if arguments.config is not None
        else os.environ.get("OPENECON_PLUGIN_CONFIG")
    )
    try:
        config_path = (
            _absolute_path(selected, "configuration file")
            if selected is not None
            else Path.home() / ".config" / "openeconometrics" / "codex-plugin.json"
        )
        launch = configured_launch(config_path)
        if os.name == "nt":
            return _run_windows(launch)
        # The MCP client owns this process and its inherited stdin/stdout. Replacing
        # it preserves EOF, signals and exit status without another protocol relay.
        os.execv(launch[0], launch)
    except ConfigurationError as error:
        print(f"OpenEconometrics plugin: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("OpenEconometrics plugin: The configured runtime could not start.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
