#!/usr/bin/env bash
# Run every baseline reported for the primary MoleculeACE regression benchmark.
# Each command uses only the official training pool; no test result is used for
# selection.  Outputs are intentionally separate so every family can be
# inspected or re-run independently.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

PYTHONPATH=src python scripts/run_cross_dataset_regression_baselines.py \
  --collections moleculeace \
  --models linear_ecfp svm_ecfp hist_gradient_boosting_ecfp mlp_ecfp extratrees_ecfp random_forest_ecfp \
  --output outputs/moleculeace_classical_baselines/full.csv

PYTHONPATH=src:scripts python scripts/run_graph_cross_dataset_baselines.py \
  --task moleculeace \
  --models chemprop gine_residual gine_nodenorm gine_pairnorm graphcliff \
  --device "${DEVICE:-cuda:0}" --epochs "${EPOCHS:-80}" \
  --output outputs/moleculeace_graph_baselines/full.csv

PYTHONPATH=src python scripts/evaluate_mtpnet_fixed_epoch.py \
  --device "${DEVICE:-cuda:0}" \
  --output-dir outputs/moleculeace_mtpnet_epoch50_audit
