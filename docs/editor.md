# Python editor assistance

The code editor provides documentation, completion and parameter hints without
executing the script or contacting a remote language service. Its appearance
follows the white/navy workbench.

| Action | Behavior |
|---|---|
| Hover over a supported function | Show the signature and description. |
| F1 with the cursor on a function | Open the same documentation using the keyboard. |
| Type inside a function call | Show its signature with the active parameter emphasized. |
| Start an argument or type `=` | Suggest unused parameters or supported literal values for a known call. |
| Ctrl+Space | Request completion at the cursor. |
| Enter while the completion menu is open | Insert the selected suggestion; normal undo remains available. |
| Escape | Dismiss the current help or completion popup. |
| Cmd/Ctrl+Enter | Run the selected code or current line in the local console. The browser team page has no Run command; team projects run in the desktop app. |
| Cmd/Ctrl+Shift+Enter | Run the complete script. |

Informational function documentation closes after eight seconds. Parameter hints
close after eight seconds without source or cursor input; typing renews that
period. Escape, losing focus, running code or replacing the document also closes
editor help. Completion menus and inline suggestions retain their normal
acceptance and dismissal behavior.

OpenEconometrics import aliases, supported `from` imports, Python builtins and user-defined
functions in the current document receive assistance. Function docstrings are
displayed as plain text. Dataframe and result methods appear only when the editor
can infer the receiver from supported expressions. OLS-specific methods are
associated with OLS results, rather than every model result.

Parameter suggestions follow the actual signature and omit keywords already
supplied elsewhere in the call. Supported OLS and binary-model values are read
from the published validation contracts. Literal defaults and boolean values are
also offered; existing quote styles are preserved when completing a string.
Local variables take precedence over unrelated builtins in argument values.

The bundled API catalog is generated from Python source with
`scripts/generate_editor_api.py`. Its signatures describe the published code;
the generator reads Python syntax rather than importing or running model modules.
Editor assistance tolerates incomplete calls and ignores strings and comments.

The completion menu is static assistance for the current script. It does not import installed
packages, inspect live Python variables or guarantee arbitrary third-party type
inference. Statistical model coverage and execution permissions are unchanged.

## Local inline suggestions

The Mac desktop edition also offers optional inline continuations under
**Suggestions**. Install the fixed Qwen2.5-Coder 0.5B Q8_0 model once, then enable
the switch. It is disabled by default. Model weights occupy 531,068,128 bytes;
they are stored separately from project files and are never synchronized to the
cloud. Disabling suggestions stops the owned worker and releases its memory.

After a short typing pause, generated code appears as muted ghost text. **Tab**
accepts visible ghost text; **Escape** dismisses it. The normal completion menu
retains priority over ghost text. Acceptance is a normal, undoable edit and never
runs Python. Moving the cursor, switching files/projects, running code, losing
edit access, closing the editor or disabling suggestions cancels pending work.

Only bounded source around the cursor reaches an authenticated loopback
llama.cpp worker. There are no external inference requests, API keys, package
imports, dataset reads or agent tools in this path. The small model is intended
for short continuations and can make mistakes; existing numerical model
implementations and execution permissions are unchanged.

The engine release, model revision, byte counts, SHA-256 checksums and licenses
are pinned in `desktop/scripts/prepare_local_suggestions.py`. Engine resources
are prepared before compilation; Settings verifies the model before loading it.
The model uses Apache-2.0 and llama.cpp uses MIT; license texts are bundled from
`desktop/licenses/`. No Ollama installation or background system service is
required. This release supports Apple Silicon Macs; Windows packaging remains deferred.
