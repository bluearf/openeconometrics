"""Run in a disposable installed desktop project's editor; no cloud operations.

The synthetic worker refuses network connections while importing and computing.
This checks offline calculation, not a physically disconnected OS/network. It
writes only qa-source.csv and qa-model.json in the current synthetic workspace.
"""

from pathlib import Path
import json
import os
import socket
import sys


assert getattr(sys, "frozen", False), "Use the installed bundled worker."
bundle = Path(sys._MEIPASS).resolve()
assert ".app/Contents/Resources/runtime/" in str(bundle)
assert not any("/Documents/GitHub/" in p or "/.codex/worktrees/" in p for p in sys.path)
connect = socket.socket.connect
attempts = []


def deny_connection(_socket, address):
    attempts.append(type(address).__name__)
    raise OSError("Network is unavailable inside the disposable QA worker.")


socket.socket.connect = deny_connection
try:
    import torch
    import openecon as oe
    from openecon.models import ResultBundle

    assert Path(torch.__file__).resolve().is_relative_to(bundle)
    assert Path(oe.__file__).resolve().is_relative_to(bundle)
    torch.set_num_threads(1)
    frame = oe.example()
    frame.to_csv("qa-source.csv", index=False)
    data = oe.read("qa-source.csv")
    model = oe.ols(data=data, y="wage", x=["education", "experience"], covariance="HC3")
    assert model.nobs == 480 and len(model.covariance_matrix) == 3
    saved = model.model_dump_json()
    Path("qa-model.json").write_text(saved)
    readback = ResultBundle.model_validate_json(Path("qa-model.json").read_text())
    assert readback.model_dump(mode="json") == model.model_dump(mode="json")
    assert readback.to_latex() == model.to_latex()
    # Optional installed overlay is verified after the explicit package command.
    import humanize

    assert humanize.__version__ == "4.14.0"
    assert ".packages" in str(Path(humanize.__file__).resolve())
    assert humanize.intcomma(12345) == "12,345"
    globals()["display"](model)
    assert attempts == []
    print(json.dumps({
        "frozen": True, "python_version": sys.version.split()[0],
        "sdk_version": oe.__version__, "torch_version": torch.__version__,
        "bundle_relative_imports": True, "system_python_uv_node_checkout_required": False,
        "inherited_path": os.environ.get("PATH"),
        "worker_network_connections_denied": True, "outbound_connection_attempts": len(attempts),
        "nobs": model.nobs, "covariance": model.spec.covariance,
        "full_json_and_latex_roundtrip": True,
        "installed_overlay": {"name": "humanize", "version": humanize.__version__},
        "scope": "Synthetic local CPU calculation; no cloud execution or physical OS disconnection."},
        sort_keys=True))
finally:
    socket.socket.connect = connect
