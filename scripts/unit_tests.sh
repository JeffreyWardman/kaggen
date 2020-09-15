#!/bin/bash
# Run unittests

set -eo

N_WORKERS=${N_TESTING_WORKERS:-1}

export LANG=en_AU.utf8

pip install -r requirements.txt
pytest -n=${N_WORKERS} -v
