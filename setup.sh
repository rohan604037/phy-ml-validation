#!/usr/bin/env bash
# One-time setup on macOS / Linux: creates a virtual environment and installs everything.
set -e
cd "$(dirname "$0")"
PY=python3
command -v $PY >/dev/null 2>&1 || PY=python
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "Python 3.9 or newer is required"; exit 1; }
$PY -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
echo
echo "Setup complete. Now run:  ./run.sh all --quick"
