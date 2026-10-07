"""Bundled uv discovery and a read-only embedded Python discovery protocol.

uv queries its chosen interpreter with a specific ``-I -B -c`` script. Frozen
OpenEconometrics is an application rather than a general Python executable, so it answers
that one query directly. It never executes the submitted script, imports the
temporary query package, downloads Python, or changes the bundled environment.
The protocol follows Astral's uv 0.9.26 ``get_interpreter_info.py``.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig

from openecon.project_packages import PackageError, installer_environment


def uv_executable() -> Path:
    """Use the included binary in releases; discover uv only in source runs."""
    name = "uv.exe" if os.name == "nt" else "uv"
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal"))
        candidate = root / "tools" / name
    else:
        candidate = None
        if importlib.util.find_spec("uv") is not None:
            import uv
            try:
                candidate = Path(uv.find_uv_bin())
            except (FileNotFoundError, RuntimeError):
                pass
        if candidate is None:
            found = shutil.which(name)
            candidate = Path(found) if found else None
    if (candidate is None or not candidate.is_absolute() or candidate.is_symlink()
            or not candidate.is_file() or not os.access(candidate, os.X_OK)):
        raise PackageError("UV_UNAVAILABLE", "The uv installer is unavailable in this OpenEconometrics runtime.")
    return candidate


def uv_environment() -> dict[str, str]:
    """Do not inherit uv settings, credentials, environment selectors or hooks."""
    result = installer_environment()
    result.update({"UV_NO_CONFIG": "1", "UV_PYTHON_DOWNLOADS": "never",
                   "UV_HTTP_TIMEOUT": "15", "UV_HTTP_RETRIES": "1",
                   "UV_CONCURRENT_DOWNLOADS": "4", "UV_CONCURRENT_INSTALLS": "4"})
    # uv's macOS wheel target must reflect the machine running embedded Python.
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        if version:
            result["MACOSX_DEPLOYMENT_TARGET"] = version
    return result


def _interpreter_info() -> dict:
    from packaging.markers import default_environment
    architecture = sysconfig.get_platform().rsplit("-", 1)[-1]
    if sys.platform == "darwin":
        parts = platform.mac_ver()[0].split(".")
        if len(parts) < 2:
            raise PackageError("UV_INTERPRETER_FAILED", "The embedded Python platform could not be identified.")
        operating_system = {"name": "macos", "major": int(parts[0]), "minor": int(parts[1])}
        architecture = platform.mac_ver()[2] or architecture
    elif os.name == "nt":
        operating_system = {"name": "windows"}
        architecture = {"win32": "i386", "amd64": "x86_64", "arm64": "aarch64"}.get(
            architecture.lower(), architecture)
    elif sys.platform.startswith("linux"):
        from packaging._manylinux import _get_glibc_version
        from packaging._musllinux import _get_musl_version
        musl = _get_musl_version(sys.executable)
        glibc = _get_glibc_version()
        if musl is not None:
            operating_system = {"name": "musllinux", "major": musl.major, "minor": musl.minor}
        elif glibc != (-1, -1):
            operating_system = {"name": "manylinux", "major": glibc[0], "minor": glibc[1]}
        else:
            raise PackageError("UV_INTERPRETER_FAILED", "The embedded Python platform could not be identified.")
    else:
        raise PackageError("UV_INTERPRETER_FAILED", "uv packages are supported on macOS, Windows and Linux.")
    architecture = {"arm64": "aarch64", "AMD64": "x86_64"}.get(architecture, architecture)
    paths = sysconfig.get_paths()
    scheme = {key: paths[key] for key in ("purelib", "platlib", "include", "scripts", "data")}
    # Only resolution and --target installation use this probe. Virtualenv paths
    # are declared for uv's schema and never used to create another environment.
    relative = ({"purelib": "Lib/site-packages", "platlib": "Lib/site-packages",
                 "include": "Include", "scripts": "Scripts", "data": ""}
                if os.name == "nt" else
                {"purelib": f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
                 "platlib": f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
                 "include": f"include/python{sys.version_info.major}.{sys.version_info.minor}",
                 "scripts": "bin", "data": ""})
    return {"result": "success", "markers": default_environment(),
            "sys_base_prefix": sys.base_prefix, "sys_base_exec_prefix": sys.base_exec_prefix,
            "sys_prefix": sys.prefix, "sys_base_executable": sys.executable,
            "sys_executable": sys.executable, "sys_path": list(sys.path),
            "site_packages": sorted({scheme["purelib"], scheme["platlib"]}),
            "stdlib": paths["stdlib"], "standalone": False, "scheme": scheme,
            "virtualenv": relative, "platform": {"os": operating_system, "arch": architecture},
            "manylinux_compatible": operating_system["name"] in {"manylinux", "musllinux"},
            "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "debug_enabled": bool(sysconfig.get_config_var("Py_DEBUG")),
            "pointer_size": "64" if sys.maxsize > 2**32 else "32"}


def maybe_uv_python_probe(arguments: list[str]) -> bool:
    """Recognize only uv's interpreter query and print data without evaluating it."""
    if len(arguments) != 4 or arguments[:3] != ["-I", "-B", "-c"]:
        return False
    script = arguments[3]
    prefix = "import sys; sys.path = ["
    suffix = "] + sys.path; from python.get_interpreter_info import main; main()"
    if (not isinstance(script, str) or len(script) > 4096
            or not script.startswith(prefix) or not script.endswith(suffix)):
        return False
    try:
        path = ast.literal_eval(script[len(prefix):-len(suffix)])
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False
    if not isinstance(path, str) or "\x00" in path or not Path(path).is_absolute():
        return False
    print(json.dumps(_interpreter_info(), ensure_ascii=True, allow_nan=False), flush=True)
    return True
