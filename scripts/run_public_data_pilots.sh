#!/usr/bin/env bash
set -euo pipefail

echo "Running public-data validation pilots..."

python -m fairprivacysignal.movielens_marketplace_validation
python -m fairprivacysignal.education_student_performance_validation

echo "Public-data validation pilots completed."
echo "Key outputs:"
echo "- docs/movielens_marketplace_validation.md"
echo "- docs/assets/movielens_marketplace_validation.svg"
echo "- outputs/tables/movielens_marketplace_validation_summary.csv"
echo "- docs/education_student_performance_validation.md"
echo "- docs/assets/education_student_performance_validation.svg"
echo "- outputs/tables/education_student_performance_validation_summary.csv"
