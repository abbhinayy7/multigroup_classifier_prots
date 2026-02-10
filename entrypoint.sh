#!/usr/bin/env bash
set -euo pipefail

# Default entrypoint for the multigroup classifier image.
# If no arguments are provided, run the default test script. Otherwise execute provided command.

if [ "$#" -eq 0 ]; then
  if [ -f "./test_binary_vs_multigroup.py" ]; then
    python ./test_binary_vs_multigroup.py
  else
    echo "No default test script found. Start a shell or provide a command to run."
    exec /bin/bash
  fi
else
  exec "$@"
fi
