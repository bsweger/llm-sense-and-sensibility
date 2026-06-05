#!/usr/bin/env bash
# Export the Streamlit Community Cloud requirements file from uv.lock.
#
# uv.lock is the source of truth, but Cloud's bundled uv is too old to parse it,
# so Cloud reads this file instead (it sits in the entrypoint directory, which
# Cloud searches before the repo root). The project is excluded (--no-emit-project)
# because Cloud can't build it (no git metadata); demos.py puts src/ on sys.path
# instead. The PyTorch CPU index is prepended because uv export does not emit index
# URLs (astral-sh/uv#10008).
set -euo pipefail

out="src/llm_sas/demos/requirements.txt"

{
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  uv export --no-dev --no-emit-project --no-hashes --format requirements-txt
} >"$out"
