# Execution output order

OpenEcon Desktop 0.3.1 fixes printed text appearing before earlier displayed
tables and models. Previously, the worker collected all printed text in one
`stdout` string and the interface placed that string above every structured
output. Charts appeared only in their separate tab.

New executions retain a bounded `events` timeline. Text segments reference their
actual position; output events reference the existing table, model or chart by
index, without duplicating its data. All output indices occur exactly once in
order, and the text segments reconstruct the unchanged aggregate `stdout`.
Worker messages and cloud archives validate the same contract. At most 41 events
accompany the existing 20-display limit.

The Results view now includes charts in this timeline. The Charts tab remains a
filtered chart view. Selecting one output hides unrelated printed text. LaTeX
source, copying and download use the same timeline and escape printed text.
The Log tab retains the aggregate text for diagnostics.

Older saved records do not contain execution-order metadata. They remain
readable with their previous layout; rerunning code in the new version records
its order. This change does not execute old scripts, rewrite history, or alter
Python's explicit `display(...)` and final-expression behavior.

Validation on 2026-10-02:

- Full Python suite: 1,521 passed, 3 platform-specific skips.
- Frontend: 84 tests and production build passed.
- An isolated local UI rendered text, table, text, model, chart, text in order;
  the LaTeX view retained that order, and the Charts filter excluded other output.
- Real console tests cover durable history, stderr before errors and truncated
  text/display limits. Cloud tests cover archived timeline readback and rejection
  of incomplete, duplicate or malformed events.

The final Apple Silicon installer is
`desktop/build/releases/0.3.1/target/release/bundle/dmg/OpenEcon_0.3.1_aarch64.dmg`
(266,281,729 bytes). Its SHA-256 is
`63fd341342eab8e6cba0bb353cdce211abdc08ebd49102c8b5e2423444b2a7e4`.
Mounted-DMG tests passed alternating print/table/model/chart order, the UTF-8
truncation boundary, publication LaTeX and persistent variables. The native smoke
test passed CPU OLS and stable-origin checks. Runtime readiness measured 0.834
seconds initially and 0.604 seconds on repeat, with the OS disk cache retained;
these are runtime startup measurements, not full project-opening timings.
The package requires macOS 15+ and has a verified local ad hoc signature, without
Developer ID signing or Apple notarization. All 4,251 files and aliases in the
previously running application remained unchanged; its UI and session were not
accessed.

Preview and production authenticated archive checks retained the exact timeline
from locally computed table/OLS/chart output, preserved its LaTeX and passed
idempotent retry. Neither check invoked cloud computation. Production now serves
`openecon-00011-w27` with 100% traffic and control image
`sha256:e1a4b1d91494af62b968001c100bde6673663d052ca0fd6bf86cf82a90f00315`.
The homepage and desktop-login configuration returned HTTP 200.
Private compute remains `openecon-sandbox-00003-qf7`; minimum instances remain
zero for both services. This release verifies desktop execution and cloud
archiving; the unchanged private compute image still produces legacy records.

Local verification records are in
`desktop/artifacts/release-0.3.1/verification.json` and
`artifacts/verification/output-order-{validation,cloud-preview,cloud-production,deployment}.json`.
