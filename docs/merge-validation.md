# Merge validation

The bounded local gate uses Python 3.11 and 3.13, checks the complete source with
Ruff, checks generated capability and editor catalogues, runs the web suite and
build, exercises the explicitly listed SDK regression modules, and builds both
Python distributions. It rejects empty, skipped, failed and timed-out test runs.
Scientific validation, physical scale tests, installed Mac acceptance and live
cloud authentication remain separate release evidence.

```sh
UV_PROJECT_ENVIRONMENT=.venv311 uv sync --frozen --python 3.11 --extra cloud --extra desktop
UV_PROJECT_ENVIRONMENT=.venv313 uv sync --frozen --python 3.13 --extra cloud --extra desktop
.venv313/bin/python scripts/verify_merge_candidate.py \
  --python .venv311/bin/python --python .venv313/bin/python \
  --directory artifacts/merge-validation/unique-run
```

For a reviewed, committed same-repository PR, append
`--publish-repo bluearf/openecon --pr NUMBER`. This publishes the stable status
`OpenEconometrics / local merge gate` only after executing the complete gate.
It rejects dirty source, another PR head, an old base, and source/base changes
during validation. Evidence includes the tested head/base and hashes of logs.
Never submit a success status manually or replay another commit's receipt.

## Hosted merge gate

`Verify OpenEconometrics` runs on pull requests, pushes to `main`, merge-group
events and manual dispatch. Its stable check name is
`OpenEconometrics / merge gate`. A standard Ubuntu runner installs the locked
Python 3.11 and 3.13 environments and runs the same bounded verifier above.
The hosted job has only read access to repository contents and does not publish
credential-backed commit statuses. Pull requests test GitHub's candidate merge
commit; the artifact records that commit, the event and the PR head/base.
It selects the runner's installed Google Chrome using
`OPENECON_BROWSER_EXECUTABLE` for genuine browser export tests, retaining Chrome's
sandbox. An invalid configured executable fails rather than selecting a wrapper.

The job retains execution identity, verifier receipts, logs, JUnit reports and
distribution artifacts for 14 days, including evidence produced before a failure.
Both Python environments, every named test module, the web suite, catalogues,
lint and package build must pass. Empty, skipped, failed and timed-out test runs
are failures. Scientific validation, physical scale, GPU hardware, installed Mac
acceptance and live cloud authentication still require their separate protocols.

## Required enforcement is a separate account gate

A successful job does **not** establish mandatory enforcement by itself. Require
`OpenEconometrics / merge gate` in an active ruleset for `main`, bind its source
to the GitHub Actions app, require the candidate to be up to date, and leave the
bypass list empty so administrators follow the same rules. Require a pull request
and block deletion and force pushes. Avoid a required local status that any writer
can publish using their own credential.

MARKET-514 stays open until actual PRs prove that missing, failed and stale checks
prevent merges and a freshly passing candidate is accepted. Use disposable base
branches to exercise rejection and successful merge without changing scientific
source on `main`; apply the same required-check policy and retain API readback and
run evidence. Read back the final ruleset, including administrator bypass, after
the probes. Keep active branches and unique work intact.

The selected SDK modules are recorded in `scripts/verify_merge_candidate.py`;
they are a regression gate, not evidence that every statistical method, Python
minor, GPU or installed application was validated. The full scientific suite and
Mac release protocols must run independently on a fixed candidate.
