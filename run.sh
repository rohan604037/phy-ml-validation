#!/usr/bin/env bash
# Run any PHYVAL command inside the virtual environment, e.g.
#   ./run.sh all --quick
#   ./run.sh run-build --fault cfo_uncorrected --severity 0.6 --budget 0.3
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first"; exit 1; }
. .venv/bin/activate
if [ $# -eq 0 ]; then set -- all; fi
python -m phyval "$@"
