#!/usr/bin/env bash
set -euo pipefail

echo "Running regression tests..."
python -m pytest -q

echo "Checking Python compilation..."
python -m compileall -q fairprivacysignal tests

echo "Running full reproducible benchmark..."
bash scripts/run_benchmark.sh

echo "Benchmark verification completed."
