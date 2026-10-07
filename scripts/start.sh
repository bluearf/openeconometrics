#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if ! command -v uv >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  printf '%s\n' 'OpenEconometrics needs uv and Node.js/npm. See README.md for installation links.' >&2
  exit 1
fi
uv sync --frozen --extra app
npm --prefix web ci
npm --prefix web run build
printf '%s\n' 'OpenEconometrics: http://127.0.0.1:8765 — stop with Ctrl+C.' >&2
exec uv run --frozen --extra app openecon serve --workspace "$project_root/.openecon" "$@"
