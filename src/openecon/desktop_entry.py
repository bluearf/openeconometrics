"""Native-shell entry point: bind an ephemeral loopback port and announce it."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import socket
import sys
import threading


def _isolate_shutdown_input() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_std_handle = kernel32.SetStdHandle
    set_std_handle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    set_std_handle.restype = ctypes.c_int
    get_std_handle = kernel32.GetStdHandle
    get_std_handle.argtypes = [ctypes.c_uint32]
    get_std_handle.restype = ctypes.c_void_p
    # Keep Python's existing stdin stream and its CRT file descriptor open for
    # the shell shutdown reader. Clear only the process default inherited by
    # new workers: CPython's buffered-stdin initialization queries the pipe's
    # position, which blocks behind the server's pending shutdown read.
    std_input_handle = ctypes.c_uint32(-10).value
    if not set_std_handle(std_input_handle, None):
        raise ctypes.WinError(ctypes.get_last_error())
    if get_std_handle(std_input_handle) is not None:
        raise OSError("The desktop shutdown input handle could not be isolated.")


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    effective_argv = sys.argv[1:] if argv is None else argv
    from openecon.uv_runtime import maybe_uv_python_probe
    if maybe_uv_python_probe(effective_argv):
        return 0
    if effective_argv and effective_argv[0] == "--package-installer":
        from openecon.package_installer import main as install_packages
        return install_packages(effective_argv[1:])
    if effective_argv and effective_argv[0] == "mcp":
        # A frozen application is its own Python executable. STDIO must never
        # enter the loopback server's readiness or stdin shutdown protocol.
        parser = argparse.ArgumentParser(description="OpenEconometrics project MCP server")
        parser.add_argument("--workspace", type=Path, required=True)
        arguments = parser.parse_args(effective_argv[1:])
        from openecon.mcp_server import create_mcp_server
        create_mcp_server(arguments.workspace).run(transport="stdio")
        return 0
    parser = argparse.ArgumentParser(description="OpenEconometrics desktop local Python runtime")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qa-metrics", action="store_true", help="Opt-in synthetic QA timing recorder (normal shell builds never enable it).")
    arguments = parser.parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    from openecon.optional_dependencies import require_extra
    require_extra("desktop", "fastapi", "uvicorn", "multipart")
    import os
    from openecon.desktop_runtime import create_desktop_app, worker_environment
    environment = worker_environment()
    os.environ.clear()
    os.environ.update(environment)
    shutdown_stream = sys.stdin.buffer
    _isolate_shutdown_input()
    import uvicorn

    app = create_desktop_app(arguments.data_root, qa_metrics=arguments.qa_metrics)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif os.name != "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", arguments.port))
        listener.listen(128)
        port = listener.getsockname()[1]

        class DesktopServer(uvicorn.Server):
            async def startup(self, sockets=None):
                await super().startup(sockets=sockets)
                if self.started:
                    descriptor = {"type": "ready", "protocol_version": 1,
                                  "url": f"http://127.0.0.1:{port}", "port": port,
                                  "token": app.state.token, "pid": os.getpid()}
                    print(json.dumps(descriptor, separators=(",", ":")), flush=True)

        server = DesktopServer(uvicorn.Config(app, host="127.0.0.1", port=port,
                                              access_log=False, log_level="warning",
                                              timeout_graceful_shutdown=2))

        def shutdown_input():
            # The pipe belongs to the native shell. No HTTP shutdown endpoint
            # accepts this protocol, and oversized/malformed lines are ignored.
            while True:
                try:
                    line = shutdown_stream.readline(4097)
                except (OSError, ValueError, AttributeError):
                    return
                if not line:
                    # A native-shell crash/force-quit closes its pipe too.
                    # Do not leave an orphaned loopback service or worker.
                    app.state.desktop_projects.close()
                    server.should_exit = True
                    return
                if len(line) > 4096 or not line.endswith(b"\n"):
                    continue
                try:
                    command = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if command == {"type": "shutdown"}:
                    app.state.desktop_projects.close()
                    server.should_exit = True
                    return

        threading.Thread(target=shutdown_input, name="OpenEconometrics shell shutdown", daemon=True).start()
        server.run(sockets=[listener])
    finally:
        app.state.desktop_projects.close()
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
