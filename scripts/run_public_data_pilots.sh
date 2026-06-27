#!/usr/bin/env bash
set -euo pipefail

echo "Running public-data validation pilots..."

python -m fairprivacysignal.movielens_marketplace_validation
python -m fairprivacysignal.education_student_performance_validation
python -m fairprivacysignal.finance_german_credit_validation
python -m fairprivacysignal.healthcare_wdbc_validation
python -m fairprivacysignal.public_services_adult_validation

echo "Public-data validation pilots completed."
echo "Key outputs:"
echo "- docs/movielens_marketplace_validation.md"
echo "- docs/assets/movielens_marketplace_validation.svg"
echo "- docs/assets/movielens_marketplace_recovery_profile.svg"
echo "- outputs/tables/movielens_marketplace_validation_summary.csv"
echo "- docs/education_student_performance_validation.md"
echo "- docs/assets/education_student_performance_validation.svg"
echo "- docs/assets/education_student_performance_recovery_profile.svg"
echo "- outputs/tables/education_student_performance_validation_summary.csv"
echo "- docs/finance_german_credit_validation.md"
echo "- docs/assets/finance_german_credit_validation.svg"
echo "- docs/assets/finance_german_credit_recovery_profile.svg"
echo "- outputs/tables/finance_german_credit_validation_summary.csv"
echo "- docs/healthcare_wdbc_validation.md"
echo "- docs/assets/healthcare_wdbc_validation.svg"
echo "- docs/assets/healthcare_wdbc_recovery_profile.svg"
echo "- outputs/tables/healthcare_wdbc_validation_summary.csv"
echo "- docs/public_services_adult_validation.md"
echo "- docs/assets/public_services_adult_validation.svg"
echo "- docs/assets/public_services_adult_recovery_profile.svg"
echo "- outputs/tables/public_services_adult_validation_summary.csv"
