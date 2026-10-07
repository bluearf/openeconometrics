"""Copyable MCP launch commands for the actual source or bundled executable."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys


def connection_config(workspace: Path) -> dict:
    command = sys.executable
    args = ([] if getattr(sys, "frozen", False) else ["-m", "openecon.cli"])
    args += ["mcp", "--workspace", str(workspace)]
    launch = [command, *args]
    quoted = subprocess.list2cmdline(launch) if os.name == "nt" else shlex.join(launch)
    return {
        "mcp_available": True,
        "mcp_server": {"command": command, "args": args},
        "mcp_command": quoted,
        "codex_command": f"codex mcp add openecon -- {quoted}",
        "claude_command": f"claude mcp add --transport stdio openecon -- {quoted}",
    }
