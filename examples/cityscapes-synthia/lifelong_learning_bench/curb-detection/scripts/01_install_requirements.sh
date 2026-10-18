#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 REQUIREMENTS_FILE" >&2
    exit 2
fi

"${PYTHON:-python}" -m pip install -r "$1"
