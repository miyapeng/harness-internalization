#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Legacy filename, now a thin entrypoint into the independent project trainer.
# Uses --request/--response; no Hydra/analyzer/parquet or patched actor is loaded.
cd "$PROJECT_ROOT"
exec python -m internalization.training.entrypoint train "$@"
