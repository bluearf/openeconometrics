"""Build a self-contained, one-directory CPU runtime without startup extraction.

Run with an environment containing openecon and PyInstaller. --python may select
the existing development environment; build-time tools never ship as requirements.
"""
from __future__ import annotations
import argparse
import base64
import csv
import importlib.metadata
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile

DESKTOP = Path(__file__).resolve().parents[1]
ROOT = DESKTOP.parent


def run_pyinstaller(command: list[str]) -> None:
    """Keep --clean and native binary rewriting inside this build's cache."""
    build = DESKTOP / "build"
    build.mkdir(parents=True, exist_ok=True)
    # Independent worktrees still share PyInstaller's default user cache.
    # --clean can remove it while another build reads or signs a cached binary.
    # Never clean that shared directory, including an inherited cache override.
    with tempfile.TemporaryDirectory(prefix="pyinstaller-config-", dir=build) as cache:
        environment = dict(os.environ, PYINSTALLER_CONFIG_DIR=cache)
        launch = command
        if sys.platform == "win32":
            # Windows CreateProcess limits the entire command line to 32,767
            # characters. Keep every frozen-module option in an owned UTF-8
            # file and pass the list directly to PyInstaller inside the child.
            # This changes only argument transport, not collection or flags.
            if command[1:3] != ["-m", "PyInstaller"]:
                raise ValueError("Expected the Python PyInstaller module invocation")
            arguments = Path(cache) / "arguments.json"
            arguments.write_text(json.dumps(command[3:]), encoding="utf-8", newline="")
            launcher = (
                "import json,sys; from pathlib import Path; "
                "from PyInstaller.__main__ import run; "
                "run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))"
            )
            launch = [command[0], "-c", launcher, str(arguments)]
        subprocess.run(launch, check=True, cwd=ROOT, env=environment)


def stdlib_inspection_sources() -> dict[str, Path]:
    """Retain stdlib source inspected by Torch's lazy JVP JIT compilation.

    PyInstaller freezes enum as bytecode. Torch's forward-AD decompositions
    inspect Enum methods at runtime, requiring the matching enum.py beside it.
    """
    import enum
    import inspect

    filename = inspect.getsourcefile(enum.Enum._generate_next_value_)
    source = Path(filename).resolve() if filename else None
    if source is None or source.name != "enum.py" or not source.is_file():
        raise RuntimeError("The desktop build requires the selected interpreter's enum.py source")
    return {"enum": source}


def inspection_source_manifest(directory: Path, sources: dict[str, Path]) -> dict:
    """Fail on omitted/stale JIT sources and bind their exact bundled bytes."""
    manifest = {}
    for name, source in sources.items():
        relative = Path("_internal") / source.name
        target = directory / relative
        payload = source.read_bytes()
        if not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError(f"Bundled {name} source differs from the selected interpreter")
        manifest[name] = {"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload)}
    return manifest


def uv_bundle_inputs() -> tuple[Path, list[tuple[Path, str]], dict]:
    """Bundle the pinned native installer and its redistribution licenses."""
    import uv
    distribution = importlib.metadata.distribution("uv")
    binary = Path(uv.find_uv_bin()).resolve()
    if not binary.is_file():
        raise ValueError("The desktop build requires the uv native executable")
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True)
    if not result.stdout.startswith("uv " + distribution.version + " "):
        raise ValueError("The uv executable and Python distribution versions differ")
    licenses = []
    for item in distribution.files or []:
        if ".dist-info/licenses/" in str(item):
            path = Path(distribution.locate_file(item))
            if path.is_file():
                licenses.append((path, "tools/licenses/uv"))
    if not licenses:
        raise ValueError("uv redistribution license files were not found")
    return binary, licenses, {"version": distribution.version,
                             "source_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                             "source_bytes": binary.stat().st_size,
                             "requires_system_python": False}

def align_openecon_metadata(directory: Path, version: str) -> Path:
    """Align copied distribution metadata with frozen source; no code is changed."""
    from packaging.version import Version
    version = str(Version(version))
    candidates = list((directory / "_internal").glob("openecon-*.dist-info"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one bundled openecon distribution")
    current = candidates[0]
    target = current.with_name(f"openecon-{version}.dist-info")
    if target != current and target.exists():
        raise ValueError("Target openecon metadata already exists")
    metadata = (current / "METADATA").read_bytes()
    boundary = re.search(rb"\r?\n\r?\n", metadata)
    end = boundary.start() if boundary else len(metadata)
    header, count = re.subn(rb"(?m)^Version: [^\r\n]*", b"Version: " + version.encode(), metadata[:end])
    if count != 1 or not re.search(rb"(?mi)^Name: openecon\r?$", header):
        raise ValueError("Invalid bundled openecon metadata")
    metadata = header + metadata[end:]
    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(metadata).digest()).decode().rstrip("=")
    rows = list(csv.reader(io.StringIO((current / "RECORD").read_text(encoding="utf-8"))))
    for row in rows:
        if len(row) != 3:
            raise ValueError("Invalid bundled openecon RECORD")
        if row[0].startswith(current.name + "/"):
            row[0] = target.name + row[0][len(current.name):]
        if row[0] == target.name + "/METADATA":
            row[1:] = [digest, str(len(metadata))]
        elif row[0] == target.name + "/RECORD":
            row[1:] = ["", ""]
    if not {target.name + "/METADATA", target.name + "/RECORD"}.issubset({row[0] for row in rows}):
        raise ValueError("Bundled openecon RECORD lacks metadata entries")
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    if current != target:
        current.rename(target)
    (target / "METADATA").write_bytes(metadata)
    (target / "RECORD").write_text(record.getvalue(), encoding="utf-8")
    return target


def prepare_distribution_notices(directory: Path) -> None:
    """Retain this checkout's grants even when copied editable metadata is stale."""
    internal = directory / "_internal"
    projects = {"openecon": (ROOT, ["LICENSE", "NOTICE", "THIRD_PARTY.md"]),
                "openecon_charts": (ROOT / "packages/openecon-charts", ["LICENSE", "NOTICE"])}
    targets = {}
    for name in projects:
        matches = list(internal.glob(name + "-*.dist-info"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one bundled {name} distribution")
        targets[matches[0]] = projects[name]
    for distribution in internal.glob("*.dist-info"):
        # Build-machine checkout URLs and cache receipts are not runtime data.
        removed = {"direct_url.json", "uv_cache.json", "uv_build.json"}
        for name in removed:
            (distribution / name).unlink(missing_ok=True)
        if distribution in targets:
            project, names = targets[distribution]
            license_root = distribution / "licenses"
            license_root.mkdir(exist_ok=True)
            for name in names:
                shutil.copyfile(project / name, license_root / name)
            # Wheel metadata is UTF-8 even when Windows uses a legacy locale.
            # Preserve its description and the bytes authenticated by RECORD.
            metadata = (distribution / "METADATA").read_text(encoding="utf-8")
            header, separator, body = metadata.partition("\n\n")
            for name in names:
                field = "License-File: " + name
                if field not in header.splitlines():
                    header += "\n" + field
            (distribution / "METADATA").write_text(
                header + separator + body, encoding="utf-8", newline=""
            )
        record_path = distribution / "RECORD"
        if not record_path.is_file():
            continue
        rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
        rows = [row for row in rows if row[0] not in {distribution.name + "/" + name for name in removed}]
        rows_by_name = {row[0]: row for row in rows}
        refreshed = [distribution / "METADATA", *sorted((distribution / "licenses").rglob("*"))]
        for file in refreshed:
            if not file.is_file():
                continue
            data = file.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows_by_name[file.relative_to(internal).as_posix()] = [file.relative_to(internal).as_posix(), "sha256=" + encoded, str(len(data))]
        rows_by_name[distribution.name + "/RECORD"] = [distribution.name + "/RECORD", "", ""]
        output = io.StringIO()
        csv.writer(output, lineterminator="\n").writerows(rows_by_name[name] for name in sorted(rows_by_name))
        record_path.write_text(output.getvalue(), encoding="utf-8", newline="")

def library_fingerprint(path: Path) -> tuple[int, str]:
    """Hash compiled Mach-O payload plus load commands, ignoring only aliases.

    PyInstaller patches each copy's LC_RPATH and code signature. Those copies
    contain identical code, but their whole-file SHA differs. Keep the root
    loader copy and make package-local copies aliases to it. All other Mach-O
    load commands and every payload byte remain part of this equivalence check.
    """
    size = path.stat().st_size
    with path.open("rb") as source:
        header = source.read(32)
        if sys.platform == "darwin" and header[:4] == b"\xcf\xfa\xed\xfe":
            ncmds, sizeofcmds = struct.unpack("<II", header[16:24])
            commands, signature_offset, first_section = [], size, size
            for _ in range(ncmds):
                command, length = struct.unpack("<II", source.read(8))
                data = source.read(length - 8)
                if command == 0x1D:  # LC_CODE_SIGNATURE
                    signature_offset = struct.unpack("<I", data[:4])[0]
                elif command not in {0x8000001C, 0x1B}:  # LC_RPATH, LC_UUID
                    commands.append(struct.pack("<II", command, length) + data)
                if command == 0x19:  # LC_SEGMENT_64
                    count = struct.unpack("<I", data[56:60])[0]
                    for index in range(count):
                        section = data[64 + index * 80:64 + (index + 1) * 80]
                        offset = struct.unpack("<I", section[48:52])[0]
                        if offset > 0:
                            first_section = min(first_section, offset)
            if not (32 + sizeofcmds <= first_section < signature_offset <= size):
                raise ValueError(f"Invalid Mach-O layout: {path}")
            digest = hashlib.sha256(header[:16] + header[24:] + b"".join(commands))
            source.seek(first_section)
            remaining = signature_offset - first_section
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Incomplete Mach-O: {path}")
                digest.update(block)
                remaining -= len(block)
            return size, digest.hexdigest()
        source.seek(0)
        return size, hashlib.file_digest(source, "sha256").hexdigest()

def optimize_runtime(directory: Path) -> dict:
    """Remove build headers and identical native-library copies, preserving paths.

    macOS loads aliases through relative symlinks. On Windows, hardlinks avoid
    requiring Developer Mode during the build; installer size is still compressed.
    """
    removed = 0
    # torch/bin/torch_shm_manager is required by torch._C initialization.
    for folder in (directory / "_internal/torch/include",):
        if folder.is_dir():
            removed += sum(item.stat().st_size for item in folder.rglob("*") if item.is_file() and not item.is_symlink())
            shutil.rmtree(folder)
    grouped: dict[tuple[int, str], list[Path]] = {}
    for item in (directory / "_internal").rglob("*"):
        if item.is_file() and not item.is_symlink() and item.suffix in {".dylib", ".dll"}:
            key = library_fingerprint(item)
            grouped.setdefault(key, []).append(item)
    saved, aliases = 0, 0
    for (size, _), paths in grouped.items():
        # Keep the root copy's @loader_path search path. Package-local copies
        # retain their filenames as aliases to this canonical loader copy.
        paths.sort(key=lambda path: len(path.parts))
        original = paths[0]
        for duplicate in paths[1:]:
            duplicate.unlink()
            if os.name == "nt":
                os.link(original, duplicate)
            else:
                duplicate.symlink_to(os.path.relpath(original, duplicate.parent))
            saved += size
            aliases += 1
    return {"removed_build_headers_bytes": removed, "deduplicated_library_bytes": saved, "library_aliases": aliases}


def econometrics_hidden_imports() -> list[str]:
    """Freeze lazy catalogue entries, forecasts and post-estimation modules.

    PyInstaller cannot follow registry import strings. Collect the whole family
    tree and fail the build if a registered entry was omitted or a package could
    not be inspected; an incomplete catalogue must never become an installer.
    """
    from PyInstaller.utils.hooks import collect_submodules
    from openecon.econometrics import registry

    modules = set(collect_submodules("openecon.econometrics", on_error="raise"))
    required = {
        "openecon.econometrics", "openecon.econometrics.registry",
        "openecon.econometrics.postest.limited_prediction",
        "openecon.econometrics.postest.ordinal_prediction",
        "openecon.econometrics.unitroot.cips_tables",
        "openecon.econometrics.tsmodels.nardl_bootstrap",
        "openecon.econometrics.core", "openecon.econometrics.postest.inference",
        "openecon.econometrics.postest.prediction",
        *[f"openecon.econometrics.{family}" for family in registry.FAMILIES],
        *[entry.entry.split(":")[0] for entry in registry.all_estimators() if not entry.legacy],
        *[target[0] for target in registry.public_exports().values()],
        *[target[0] for target in registry.forecasters().values()],
    }
    required = {name for name in required if name.startswith("openecon.econometrics")}
    missing = sorted(required - modules)
    if missing:
        raise RuntimeError("The frozen econometrics catalogue lacks required modules: " + ", ".join(missing))
    return sorted(modules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path)
    parser.add_argument("--skip-web-build", action="store_true")
    args = parser.parse_args()
    if args.python and args.python.resolve() != Path(sys.executable).resolve():
        command = [str(args.python), str(Path(__file__).resolve())]
        if args.skip_web_build:
            command.append("--skip-web-build")
        subprocess.run(command, check=True, cwd=ROOT)
        return
    if not args.skip_web_build:
        subprocess.run(["npm.cmd" if os.name == "nt" else "npm", "run", "build"], cwd=ROOT / "web", check=True)
    # The frozen runtime contains the exact local interface it serves. No CDN
    # or checkout assets are required after installation.
    from openecon import __version__
    import openecon.desktop_entry  # noqa: F401 - verify build entry is available
    import PyInstaller  # noqa: F401 - verify the selected build tool is available
    hidden = [
        "openecon.desktop_entry", "openecon.desktop_runtime", "openecon.server",
        "openecon.mcp_launcher", "openecon.mcp_server", "openecon.mcp_jobs", "mcp.server.fastmcp",
        "mcp.server.stdio",
        "openecon.console_worker", "openecon.analysis", "openecon.data", "openecon.survey",
        "openecon.project_packages", "openecon.package_installer", "openecon.script_packages",
        "openecon.package_requirements", "openecon.uv_runtime", "uv", "packaging.requirements",
        sysconfig._get_sysconfigdata_name() if os.name != "nt" else "sysconfig",
        "openecon.linear_ols", "openecon.linear_ols.spec", "openecon.linear_ols.design",
        "openecon.linear_ols.estimation", "openecon.linear_ols.postestimation", "openecon.linear_ols.streaming",
        "openecon.linear_ols.streaming_postestimation",
        "openecon.linear_ols.lags",
        "openecon.linear_ols.publication",
        "openecon.dataset", "openecon.dataset_prepare", "openecon.source_readers", "openecon.streaming_analysis",
        "openecon.streaming_design", "openecon.engines.streaming_binary",
        "openecon.engines.streaming_groups",
        "openecon.engines.torch_engine", "openecon.engines.inference",
        "openecon.engines.streaming_ols",
        "openecon.engines.separation", "openecon.engines.execution", "openecon.resources",
        "openecon.engines.streaming_filter", "openecon.engines.streaming_hac", "openecon.engines.arch_state",
        "openecon.plotting", "openecon.latex", "openecon.output_latex", "openecon.team_output",
        "openecon.networks", "openecon._network_signed", "openecon_charts.network",
        "openecon._network_store", "openecon._network_disk_algorithms",
        "openecon._network_multi",
        "openecon._network_multilayer",
        "openecon._network_dynamic", "openecon._network_dynamic_io",
        "openecon._network_ergm", "openecon._network_saom",
        "openecon._network_latent_blocks", "openecon._network_learning",
        "openecon._network_workbench", "openecon.plot_artifacts",
        "openecon._network_sparse", "openecon._network_communities", "openecon._network_centrality",
        "openecon._network_topology", "openecon._network_io",
        "openecon._network_paths", "openecon._network_flow", "openecon._network_cost_flow", "openecon._network_bipartite",
        "openecon._network_bounded_paths", "openecon._network_hypergraph",
        "openecon._network_spectral", "openecon._network_device",
        "openecon._network_cut", "openecon._network_directed_cut", "openecon._network_motifs", "openecon._network_qap",
        "openecon._network_mrqap", "openecon._network_sbm", "openecon._network_temporal",
        "openecon._network_graphlets", "openecon._network_matching",
        "openecon._network_poisson", "openecon._network_dc_sbm", "openecon._network_block_prediction",
        "openecon_charts", "openecon_charts.charts", "openecon_charts.latex", "openecon_charts.annotations", "openecon_charts.streaming", "openecon_charts.resident", "openecon_charts.network_export", "openecon_charts.timeline",
        "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
        "python_multipart", "pyarrow", "pyarrow.dataset", "openpyxl", "pyreadstat",
    ]
    hidden.extend(econometrics_hidden_imports())
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--console", "--name", "openecon-runtime", "--distpath", str(DESKTOP / "runtime"),
        "--workpath", str(DESKTOP / "build" / "pyinstaller"),
        "--specpath", str(DESKTOP / "build"), "--collect-data", "openecon",
        "--collect-data", "openecon_charts", "--collect-all", "torch", "--collect-all", "pip",
    ]
    from openecon.project_packages import PROTECTED_DISTRIBUTIONS
    uv_binary, uv_licenses, uv_metadata = uv_bundle_inputs()
    inspection_sources = stdlib_inspection_sources()
    for source in inspection_sources.values():
        command.extend(["--add-data", str(source) + os.pathsep + "."])
    command.extend(["--add-binary", str(uv_binary) + os.pathsep + "tools"])
    for license_path, destination in uv_licenses:
        command.extend(["--add-data", str(license_path) + os.pathsep + destination])
    for distribution in PROTECTED_DISTRIBUTIONS:
        try:
            importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        command.extend(["--copy-metadata", distribution])
    for module in hidden:
        command.extend(["--hidden-import", module])
    # These packages implement deployed services and are intentionally outside
    # the desktop execution environment.
    for module in ["firebase_admin", "google.cloud", "openecon.team_cloud", "openecon.cloud", "openecon.team_server", "matplotlib", "IPython", "pytest", "statsmodels", "scipy"]:
        command.extend(["--exclude-module", module])
    command.append(str(DESKTOP / "scripts" / "frozen_entry.py"))
    run_pyinstaller(command)
    runtime = DESKTOP / "runtime" / "openecon-runtime"
    align_openecon_metadata(runtime, __version__)
    prepare_distribution_notices(runtime)
    # Windows freezes the installed wheel. Its assets can predate the web build
    # performed above, so always package the freshly built interface explicitly.
    static = runtime / "_internal" / "openecon" / "static"
    if static.exists():
        shutil.rmtree(static)
    shutil.copytree(ROOT / "src" / "openecon" / "static", static)
    optimization = optimize_runtime(runtime)
    inspection_manifest = inspection_source_manifest(runtime, inspection_sources)
    packages = {}
    for name in ["openecon", "openecon-charts", "torch", "pandas", "pyarrow", "fastapi", "uvicorn", "uv", "pydantic", "PyInstaller"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # The frozen modules come from the selected source checkout; editable
    # installation metadata can retain the previous release's version.
    packages["openecon"] = __version__
    manifest = {
        "format": "pyinstaller-onedir", "openecon_version": __version__,
        "python": platform.python_version(), "platform": sys.platform,
        "architecture": platform.machine(), "compute": "cpu_with_optional_checked_gpu_factors", "packages": packages,
        "compute_precision": "float64; Metal QR is a checked preconditioner with CPU float64 refinement",
        "startup_extraction": False, "requires_system_python": False,
        "optimization": optimization,
        "tools": {"uv": uv_metadata},
        "inspection_sources": inspection_manifest,
    }
    (DESKTOP / "runtime" / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline=""
    )
    print(json.dumps({"runtime": str(DESKTOP / "runtime" / "openecon-runtime"), "manifest": manifest}))

if __name__ == "__main__":
    main()
