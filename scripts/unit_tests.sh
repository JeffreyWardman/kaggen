#!/bin/bash
# Run unit tests

set -euo pipefail
export LANG=en_AU.utf8

python3 -m pip install -r requirements.txt
black --check .
isort --check .
pytest --dead-fixtures -v
