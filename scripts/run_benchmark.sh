#!/usr/bin/env bash
set -euo pipefail

echo "Running FairPrivacySignal benchmark pipeline..."

python -m fairprivacysignal.data_generator --out data/synthetic --seed 42
python -m fairprivacysignal.sanity_checks
python -m fairprivacysignal.signal_loss
python -m fairprivacysignal.ranking
python -m fairprivacysignal.privacy_recovery
python -m fairprivacysignal.visualize_results
python -m fairprivacysignal.capacity_allocation
python -m fairprivacysignal.capacity_sensitivity
python -m fairprivacysignal.multiseed_capacity_sensitivity
python -m fairprivacysignal.score_matched_calibration
python -m fairprivacysignal.multiseed_evaluation
python -m fairprivacysignal.benchmark_overview
python -m fairprivacysignal.architecture_diagram
python -m fairprivacysignal.benchmark_validation

echo "Benchmark pipeline completed."
echo "Key outputs:"
echo "- docs/assets/"
echo "- docs/multiseed_results.md"
echo "- outputs/tables/"
