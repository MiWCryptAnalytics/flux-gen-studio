#!/usr/bin/env bash
# Launch FLUX Gen Studio.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv found. Create one first:" >&2
  echo "  python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec .venv/bin/python -m fluxstudio "$@"
