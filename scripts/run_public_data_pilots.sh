#!/usr/bin/env bash
set -euo pipefail

echo "Running public-data validation pilots..."

python -m fairprivacysignal.movielens_marketplace_validation

echo "Public-data validation pilots completed."
echo "Key outputs:"
echo "- docs/movielens_marketplace_validation.md"
echo "- docs/assets/movielens_marketplace_validation.svg"
echo "- outputs/tables/movielens_marketplace_validation_summary.csv"
