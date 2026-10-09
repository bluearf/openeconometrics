"""Offline network export through an installed Chromium, with no runtime dependencies.

The CDP transport connects only to the disposable browser's loopback endpoint.
It never uses a user's existing browser, downloads a browser or fetches fonts.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

MAX_BYTES = 128 * 1024 * 1024
MAX_MESSAGE = (MAX_BYTES * 4 // 3) + 1024 * 1024
MAX_PIXELS = 16_000_000


class NetworkExportError(RuntimeError):
    """A controlled browser, layout or export failure; no output was published."""


def _browser_environment():
    """Keep Chromium's Linux singleton socket within the AF_UNIX path limit.

    Chromium creates its own private socket directory below TMPDIR. The export
    document and explicit browser profile still use the caller-owned directory.
    A long Python temporary root must not become the native socket root.
    """
    if not sys.platform.startswith("linux"):
        return None
    environment = dict(os.environ)
    environment.update(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
    return environment


def _supervise_browser(command, state, parent, log_path, create_session):
    """Keep an owned leader alive until teardown, in Python and frozen runtimes."""
    if create_session:
        os.setsid()
    state = Path(state)
    browser = None

    def record(phase, code=None):
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "anchor_pid": os.getpid(), "pgid": os.getpgrp(),
            "browser_pid": browser.pid if browser else None,
            "phase": phase, "returncode": code,
        }))
        os.replace(temporary, state)

    try:
        if os.getppid() != parent:
            return
        record("group-started")
        # This new supervisor has no application threads. Spawn the native
        # executable with default TERM, then protect its still-owned leader.
        # No Python flags are ever passed to a frozen runtime or to Chrome.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with Path(log_path).open("ab") as log:
            browser = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, env=_browser_environment())
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            record("browser-started")
            while True:
                if os.getppid() != parent:
                    return
                try:
                    code = browser.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    continue
                record("native-exited", code)
                break
            while os.getppid() == parent:
                time.sleep(.1)
    finally:
        # Parent death or a supervisor/status-file failure cannot orphan native
        # children. This live leader still reserves its own unique group ID.
        os.killpg(os.getpgrp(), signal.SIGKILL)


class _FrozenBrowserProcess:
    """Popen-like owned handle for freeze_support's multiprocessing dispatch."""

    def __init__(self, process):
        self._process = process
        self.pid = process.pid
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            self.returncode = self._process.exitcode
            if self.returncode is not None:
                self._process.close()
        return self.returncode

    def wait(self, timeout):
        if self.returncode is None:
            self._process.join(timeout)
            code = self._process.exitcode
            if code is None:
                raise subprocess.TimeoutExpired("owned browser supervisor", timeout)
            self.returncode = code
            self._process.close()
        return self.returncode

    def kill(self):
        self._process.kill()


_PYTHON_BROWSER_BOOTSTRAP = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); "
    "from openecon_charts.network_export import _supervise_browser; "
    "_supervise_browser(json.loads(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], False)"
)


def _start_browser(command, log, state):
    """Own a reserved POSIX group identity even after the native launcher exits."""
    if os.name != "posix":
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    if getattr(sys, "frozen", False):
        # The desktop entry calls freeze_support before its app dispatcher.
        # Its executable is an application, not a Python -I/-c interpreter.
        import multiprocessing
        worker = multiprocessing.get_context("spawn").Process(
            target=_supervise_browser,
            args=(command, str(state), os.getpid(), str(log.name), True),
        )
        worker.start()
        process = _FrozenBrowserProcess(worker)
    else:
        # Ordinary library users need no multiprocessing __main__ guard. The
        # explicit package path is this installed module's root, never cwd.
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", _PYTHON_BROWSER_BOOTSTRAP,
             str(Path(__file__).resolve().parents[1]), json.dumps(command),
             str(state), str(os.getpid()), str(log.name)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
    process._openecon_group_anchor = True
    process._openecon_browser_state = state
    return process


def _browser_state(process):
    state = getattr(process, "_openecon_browser_state", None)
    if state is None:
        return None
    try:
        record = json.loads(state.read_text())
    except FileNotFoundError:
        return None
    if record["anchor_pid"] != process.pid or record["pgid"] != process.pid:
        raise NetworkExportError("Disposable browser group identity changed.")
    if record["phase"] not in {"group-started", "browser-started", "native-exited"}:
        raise NetworkExportError("Invalid disposable browser phase.")
    code = record["returncode"]
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise NetworkExportError("Invalid disposable browser exit status.")
    return record


def _browser_exit_code(process):
    """Read the native browser status without reaping the owned group leader."""
    record = _browser_state(process)
    return record["returncode"] if record else None


def _stop_browser(process):
    """Stop every owned POSIX browser child, including after its launcher exits."""
    if os.name == "posix":
        if not getattr(process, "_openecon_group_anchor", False):
            raise NetworkExportError("Disposable browser group leader is not owned.")
        # multiprocessing's global cleanup may already have reaped this child
        # while another worker started. Admit ownership only after one poll;
        # never poll/reap again between group admission and the final signal.
        if process.poll() is not None:
            # A startup poll already reaped this failed supervisor. Its finally
            # block stops its group; never mask the original startup failure or
            # signal a numeric PGID that could now identify another browser.
            return
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            process.wait(timeout=5)
            return
        if group != process.pid:
            # Frozen spawn has not entered its own session yet. Kill only this
            # unreaped owned child; signalling its inherited group is forbidden.
            process.kill()
            process.wait(timeout=5)
            return
        try:
            record = _browser_state(process)
            if record and record["phase"] != "group-started":
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while _browser_exit_code(process) is None and time.monotonic() < deadline:
                    time.sleep(.01)
        finally:
            # No poll/reap occurs between group ownership and escalation. The
            # live/unreaped leader reserves its PID until this final wait.
            # Permission failures propagate, rather than declaring cleanup.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _remove_profile(profile):
    """Finish deletion after bounded process shutdown; persistent errors fail."""
    for attempt in range(5):
        try:
            shutil.rmtree(profile)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            # A just-killed child can finish a pending filesystem operation.
            # Retry only this race, with a fixed 200 ms total delay budget.
            if exc.errno != errno.ENOTEMPTY or attempt == 4:
                raise
            time.sleep(.05)


def _browser(explicit):
    if explicit is None:
        explicit = os.environ.get("OPENECON_BROWSER_EXECUTABLE")
    if explicit is not None:
        path = Path(explicit).expanduser().resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("browser_executable must be an executable file.")
        return str(path)
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        *[shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "msedge")],
    ]
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            candidates.extend(str(Path(base) / suffix) for suffix in (
                "Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise NetworkExportError("Install Chrome, Chromium or Edge, or pass browser_executable. "
                             "Network export does not download a browser.")


def _wait_for_debugging_port(descriptor, process, deadline):
    """Read a completed Chromium descriptor within the existing startup budget."""
    browser_path = "/devtools/browser/"
    while True:
        if process.poll() is not None or _browser_exit_code(process) is not None:
            raise NetworkExportError("Disposable browser could not start.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NetworkExportError("Browser startup exceeded the export timeout.")
        try:
            lines = descriptor.read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, UnicodeDecodeError):
            lines = []
        # Chromium creates the file before writing its port and browser path.
        # The second line proves that the entire numeric port line was written.
        if (len(lines) == 2 and 1 <= len(lines[0]) <= 5 and lines[0].isdecimal()
                and lines[1].startswith(browser_path) and len(lines[1]) > len(browser_path)
                and lines[1].isprintable() and not any(char.isspace() for char in lines[1])
                and "?" not in lines[1] and "#" not in lines[1]):
            port = int(lines[0])
            if 1 <= port <= 65535:
                return port
        time.sleep(min(.05, remaining))


class _CDP:
    """Minimal RFC6455/CDP client; bounded messages, deadline and strict handshake."""

    def __init__(self, url, deadline):
        parsed = urlsplit(url)
        if parsed.scheme != "ws" or parsed.hostname != "127.0.0.1" or not parsed.port:
            raise NetworkExportError("Invalid disposable browser endpoint.")
        self.deadline, self.sequence, self.buffer = deadline, 0, bytearray()
        self.sock = socket.create_connection((parsed.hostname, parsed.port), self.remaining())
        try:
            key = base64.b64encode(os.urandom(16)).decode()
            self.sock.sendall((f"GET {parsed.path} HTTP/1.1\r\nHost: 127.0.0.1:{parsed.port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n").encode())
            while b"\r\n\r\n" not in self.buffer:
                self.sock.settimeout(self.remaining())
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise NetworkExportError("Browser closed during WebSocket handshake.")
                self.buffer.extend(chunk)
                if len(self.buffer) > 65536:
                    raise NetworkExportError("Invalid oversized browser handshake.")
            headers, rest = bytes(self.buffer).split(b"\r\n\r\n", 1)
            self.buffer = bytearray(rest)
            expected = base64.b64encode(hashlib.sha1((key +
                "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
            pairs = dict(line.split(":", 1) for line in headers.decode().split("\r\n")[1:] if ":" in line)
            pairs = {k.lower(): v.strip() for k, v in pairs.items()}
            if b" 101 " not in headers.split(b"\r\n")[0] or pairs.get("sec-websocket-accept") != expected:
                raise NetworkExportError("Browser rejected the WebSocket handshake.")
        except BaseException:
            self.sock.close()
            raise

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise NetworkExportError("Network export exceeded its timeout.")
        return remaining

    def _read(self, count):
        while len(self.buffer) < count:
            self.sock.settimeout(self.remaining())
            chunk = self.sock.recv(min(65536, count - len(self.buffer)))
            if not chunk:
                raise NetworkExportError("Disposable browser closed before export completed.")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def _send(self, payload, opcode=1):
        mask = os.urandom(4)
        size = len(payload)
        header = bytes([0x80 | opcode])
        header += (bytes([0x80 | size]) if size < 126 else
                   bytes([0xFE]) + struct.pack("!H", size) if size < 65536 else
                   bytes([0xFF]) + struct.pack("!Q", size))
        self.sock.settimeout(self.remaining())
        self.sock.sendall(header + mask + bytes(value ^ mask[i % 4] for i, value in enumerate(payload)))

    def _message(self):
        result = bytearray()
        while True:
            first, second = self._read(2)
            opcode, final = first & 15, bool(first & 128)
            length = second & 127
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            if second & 128 or length > MAX_MESSAGE - len(result):
                raise NetworkExportError("Browser response exceeded its message budget.")
            payload = self._read(length)
            if opcode == 8:
                raise NetworkExportError("Browser closed its export connection.")
            if opcode == 9:
                self._send(payload, 10)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise NetworkExportError("Unexpected browser message type.")
            result.extend(payload)
            if final:
                return json.loads(result)

    def call(self, method, **params):
        self.sequence += 1
        self._send(json.dumps(dict(id=self.sequence, method=method, params=params)).encode())
        while True:
            response = self._message()
            if response.get("id") != self.sequence:
                continue  # Bounded by the same overall deadline, including event traffic.
            if "error" in response:
                raise NetworkExportError(str(response["error"].get("message", "Browser protocol error")))
            return response["result"]


def export_network(plot, path, format, *, width=960, height=600, scale=1,
                   timeout=60, overwrite=False, browser_executable=None):
    """Render then atomically publish an explicitly requested network format."""
    if plot.kind != "network":
        raise ValueError("Direct PDF/SVG/PNG export currently requires a network PlotSpec.")
    if format not in ("pdf", "svg", "png"):
        raise ValueError("Unsupported network export format.")
    for name, value, low, high in (("width", width, 320, 2400), ("height", height, 240, 1600)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}.")
    for name, value, low, high in (("scale", scale, .5, 2), ("timeout", timeout, 1, 300)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and from {low} to {high}.")
    if format != "png" and scale != 1:
        raise ValueError("scale applies only to PNG; PDF and SVG remain vectors.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("Export destination must not be a symbolic link.")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    browser = _browser(browser_executable)
    owned = plot.with_options(width=width, height=height)
    document = owned.to_html(height=height)
    if len(document.encode()) > MAX_BYTES:
        raise NetworkExportError("Offline export document exceeds 128 MiB.")
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="openecon-network-export-") as temporary:
        root = Path(temporary)
        page, profile = root / "network.html", root / "profile"
        page.write_text(document, encoding="utf-8")
        with (root / "browser.log").open("wb") as log:
            process = _start_browser([browser, "--headless", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-component-update", "--disable-sync",
                "--disable-extensions", "--disable-crash-reporter", "--remote-debugging-address=127.0.0.1",
                "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                log, root / "browser-state.json")
            cdp = None
            try:
                descriptor = profile / "DevToolsActivePort"
                opener = build_opener(ProxyHandler({}))
                port = _wait_for_debugging_port(descriptor, process, deadline)
                with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=max(.01, deadline-time.monotonic())) as response:
                    pages = json.loads(response.read(65536))
                endpoint = next(item["webSocketDebuggerUrl"] for item in pages if item.get("type") == "page")
                cdp = _CDP(endpoint, deadline)
                cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height+300,
                         deviceScaleFactor=scale, mobile=False)
                cdp.call("Page.navigate", url=page.as_uri())
                expression = """(async () => {
                  const end = performance.now() + TIMEOUT;
                  const tick = () => new Promise(r => setTimeout(r, 16));
                  let chart;
                  while (!(chart = window.OpenEconNetworkCharts?.get(document.getElementById('chart')))) {
                    const host = document.getElementById('chart');
                    if (host?.getAttribute('role') === 'alert') throw new Error(host.textContent);
                    if (performance.now() > end) throw new Error('Chart initialization timeout');
                    await tick();
                  }
                  await document.fonts.ready;
                  if (!document.fonts.check('12px Barlow')) throw new Error('Barlow font could not load');
                  while (!chart.layoutState || chart.layoutState.status === 'pending') {
                    if (performance.now() > end) throw new Error('Layout completion timeout');
                    await tick();
                  }
                  if (chart.layoutState.status !== 'complete') throw new Error(chart.layoutState.reason || 'Layout failed');
                  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
                  chart.draw();
                  let blob;
                  if (FORMAT === 'pdf') blob = new Blob([OpenEconNetworkPDF.bytes(chart.exportScene())]);
                  else if (FORMAT === 'svg') blob = new Blob([new XMLSerializer().serializeToString(chart.snapshot())]);
                  else {
                    const layout = chart.exportLayout();
                    if (chart.canvas.width * Math.round(layout.height * chart.ratio) > MAXPIXELS)
                      throw new Error('PNG export exceeds 16 million pixels');
                    blob = await new Promise((resolve, reject) => {
                      chart.save = (value, extension) => { if (extension !== 'png') reject(new Error('Wrong export format')); else resolve(value); };
                      chart.export('png');
                      const check = () => {
                        if (chart.destroyed || /Could not export/.test(chart.status.textContent)) reject(new Error('PNG export failed'));
                        else if (performance.now() > end) reject(new Error('PNG export timeout'));
                        else if (chart.pngExport) setTimeout(check, 16);
                      };
                      check();
                    });
                  }
                  if (!blob || !blob.size || blob.size > MAXBYTES) throw new Error('Export exceeds its 128 MiB byte budget');
                  const array = new Uint8Array(await blob.arrayBuffer());
                  const parts = [];
                  for (let i = 0; i < array.length; i += 32768) parts.push(String.fromCharCode(...array.subarray(i, i+32768)));
                  return btoa(parts.join(''));
                })()""".replace("TIMEOUT", str(int(timeout*1000))).replace("FORMAT", json.dumps(format)).replace(
                    "MAXPIXELS", str(MAX_PIXELS)).replace("MAXBYTES", str(MAX_BYTES))
                result = cdp.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
                if "exceptionDetails" in result:
                    error = result["exceptionDetails"].get("exception", {}).get("description", "Network export failed")
                    raise NetworkExportError(error)
                value = result["result"].get("value")
                if not isinstance(value, str) or len(value) > MAX_BYTES*4//3+4:
                    raise NetworkExportError("Invalid or oversized browser export.")
                data = base64.b64decode(value, validate=True)
                signatures = {"pdf": b"%PDF-1.7", "png": b"\x89PNG\r\n\x1a\n", "svg": b"<svg"}
                if not data.startswith(signatures[format]) or len(data) > MAX_BYTES:
                    raise NetworkExportError("Browser returned an invalid export file.")
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                raise NetworkExportError(f"Network export failed: {exc}") from exc
            finally:
                try:
                    if cdp:
                        cdp.sock.close()
                finally:
                    _stop_browser(process)
                    _remove_profile(profile)
    # Same-directory staging, exclusive publication by default, no partial files.
    fd, staging = tempfile.mkstemp(prefix=".openecon-export-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            if destination.is_symlink():
                raise ValueError("Export destination became a symbolic link.")
            os.replace(staging, destination)
        else:
            os.link(staging, destination)
    finally:
        Path(staging).unlink(missing_ok=True)
    return destination
