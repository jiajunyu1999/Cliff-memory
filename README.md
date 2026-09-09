# PC-GAM Molecular Cliffs

This directory is an independent, leakage-controlled workspace for molecular
activity cliffs.  It keeps the original benchmark split intact and uses one
fixed seed (`42`) only for deterministic validation/model training.  Test
labels are never used for model selection.

## Current benchmark scope

- **MoleculeACE**: 30 target-specific regression datasets, evaluated by RMSE
  and cliff-only RMSE on its official test split.
- **ACNet**: 190 MMP classification tasks and its target-held-out mixed task.
- **Dablander et al.**: D2, factor Xa, and SARS-CoV-2 Mpro molecule/MMP data.
- **GraphCliff** and **ACANet**: recent lightweight model/loss references.
- **KerRead** and **ACRead**: readout references for kernel-center and
  centrality/attention mechanisms.

Official sources are pinned under `third_party/`; downloaded ACNet source data
are under `data/acnet/raw/`.  Exact revisions and file hashes are written by
`scripts/audit_assets.py`.

## First reproducible runs

```bash
cd molecular_cliffs
PYTHONPATH=src python scripts/audit_assets.py
PYTHONPATH=src python scripts/run_moleculeace.py --model svm_official --dataset all
PYTHONPATH=src python scripts/run_moleculeace.py --model calibrated_mmp --dataset all
PYTHONPATH=src python scripts/run_moleculeace_multitarget_action.py --output-dir outputs/moleculeace_multitarget_action_v1 --epochs 220 --device cuda:0
PYTHONPATH=src python scripts/run_moleculeace_multitarget_action.py --output-dir outputs/moleculeace_multitarget_action_balanced_v2 --epochs 220 --device cuda:0 --loss-mode action_balanced
PYTHONPATH=src python scripts/run_moleculeace_multiscale_kernel.py
PYTHONPATH=src:scripts python scripts/run_moleculeace_local_chemistry_kernel.py
PYTHONPATH=src:scripts python scripts/run_moleculeace_action_salience_kernel.py
PYTHONPATH=src:scripts python scripts/run_moleculeace_deshrunk_kernel.py
PYTHONPATH=src:scripts python scripts/validate_moleculeace_count_kernel.py
PYTHONPATH=src:scripts python scripts/run_moleculeace_count_kernel.py
PYTHONPATH=src:scripts python scripts/validate_moleculeace_collision_free_kernel.py
PYTHONPATH=src:scripts python scripts/run_moleculeace_collision_free_kernel.py
PYTHONPATH=src python scripts/run_moleculeace_conditional_mmp.py --device cuda:0
PYTHONPATH=src python scripts/evaluate_mtpnet_fixed_epoch.py --device cuda:0
PYTHONPATH=src python scripts/build_moleculeace_target_context.py
PYTHONPATH=src python scripts/build_action_ensemble.py
PYTHONPATH=src python scripts/build_action_ensemble.py --base-model svm_official --rbf-dir outputs/moleculeace_official_baselines --validation-gain-threshold 0.002 --shrinkage 0.8 --output-dir outputs/action_ensemble_guarded_v3
PYTHONPATH=src python scripts/run_acnet_action.py --data MMP_AC_Small.json --model logreg --output acnet_small_action_logreg_v1
PYTHONPATH=src python scripts/run_acnet_action.py --data MMP_AC_Small.json --model extratrees --output acnet_small_action_extratrees_v1
PYTHONPATH=src python scripts/run_acnet_action.py --data MMP_AC_Mixed_Screened.json --splitter target --sparse --model sgd_logreg --feature-modes endpoint action endpoint_action --output acnet_mixed_target_sgd_logreg_v1
PYTHONPATH=src python scripts/run_dablander_external.py
PYTHONPATH=src python scripts/audit_moleculeace_results.py
PYTHONPATH=src python -m pytest -q
```

Use `--dataset all` for the fixed 30-target suite.  Results are CSV/JSON files
under `outputs/`; every prediction row includes the official split and cliff
flag for independent metric recomputation.

## Locally executable baseline suite

Every baseline named in the MoleculeACE comparison has a checked-in local
entry point.  The classical ECFP family and the graph family deliberately use
fixed, global hyperparameters and train only on the official training pool.
The graph command below runs the Chemprop-style D-MPNN, residual GINE,
NodeNorm GINE, PairNorm GINE, and GraphCliff on all 30 targets.  MTPNet is
evaluated from its released, fixed final checkpoint rather than a test-selected
checkpoint.

```bash
PYTHONPATH=src python scripts/run_cross_dataset_regression_baselines.py \
  --collections moleculeace --output outputs/cross_dataset_regression_baselines/full.csv
PYTHONPATH=src:scripts python scripts/run_graph_cross_dataset_baselines.py \
  --task moleculeace --models chemprop gine_residual gine_nodenorm gine_pairnorm graphcliff \
  --device cuda:0 --epochs 80 --output outputs/moleculeace_graph_baselines/full.csv
PYTHONPATH=src python scripts/evaluate_mtpnet_fixed_epoch.py \
  --device cuda:0 --output-dir outputs/moleculeace_mtpnet_epoch50_audit
```

For a fast, identical-code-path smoke test, append
`--targets CHEMBL204_Ki --epochs 1` to the graph command; the ECFP and ACNet
launchers accept `--targets` as well.  ACNet is a separate matched-pair
classification task, so it uses the same local ECFP and graph encoder families
but is not pooled with MoleculeACE regression:

```bash
PYTHONPATH=src:scripts python scripts/run_acnet_cross_baselines.py \
  --output outputs/acnet_cross_baselines/full.csv
PYTHONPATH=src:scripts python scripts/run_graph_cross_dataset_baselines.py \
  --task acnet --models chemprop gine_residual gine_nodenorm gine_pairnorm graphcliff \
  --device cuda:0 --output outputs/acnet_graph_baselines/full.csv
```

## Current frozen result

The strongest compliant MoleculeACE candidate is one precomputed-kernel SVR.
It uses collision-free sparse substructure identities rather than a fixed-size
hash and preserves both binary presence and exact occurrence count across five
views: chiral Morgan radius 1/2/3, pharmacophore Morgan radius 2, and chiral
atom pairs.  The representation change was accepted on one fixed internal
validation split and then evaluated once on official test.  No model ensemble,
checkpoint search, hyperparameter scan, or seed search is used.

Against the reproduced official SVM baseline `0.67120 / 0.75113 / 0.62373`
(overall / cliff / non-cliff macro RMSE):

| Compliant single model | Overall | Cliff | Non-cliff | Relative gain vs SVM |
|---|---:|---:|---:|---:|
| Collision-free binary+count kernel | **0.62391** | **0.71783** | **0.57211** | **+7.04% / +4.43% / +8.28%** |
| Hashed binary+count kernel | 0.63075 | 0.72196 | 0.58020 | +6.03% / +3.88% / +6.98% |
| Local-chemistry binary kernel | 0.63990 | 0.73089 | 0.58893 | +4.66% / +2.70% / +5.58% |
| Multitarget action v1 | 0.65593 | 0.73040 | 0.61349 | +2.27% / +2.76% / +1.64% |
| Action-balanced v2 | 0.65817 | **0.72769** | 0.61776 | +1.94% / **+3.12%** / +0.96% |

A paired 5,000-draw target/molecule cluster bootstrap places the strongest
model's relative gain at 5.54%--8.55% overall and 1.77%--6.93% on cliffs (95%
interval).  None of the draws reaches the requested 10%/20% thresholds; see
`reports/results/collision_free_bootstrap.json`.

The action-balanced objective upweights train-only analog edges with large
observed activity differences.  It improves cliff RMSE on 19/30 targets versus
SVM, but the macro gains are still far below the requested +10% overall and
+20% cliff target.  The detailed audit is in
`reports/results/multitarget_action_audit.csv`.

A single equal-weight radius-1/2/3 Tanimoto kernel provides the structural
control: `0.65686 / 0.74370 / 0.60649`.  Adding chirality, pharmacophore, and
long-range atom-pair views improves this to `0.63990 / 0.73089 / 0.58893`.
Its much larger non-cliff than cliff gain isolates the open problem: resolving
chemical identity still does not recover the magnitude of local potency jumps.

A staged pose-aware follow-up used experimental holo pockets and GNINA/Vina
with one pose, seed 42, exhaustiveness 1, and CNN scoring disabled.  Residue
contact and affinity views improved both internal-validation metrics on CLK4
and JAK2, but the frozen official-test confirmation did not transfer: CLK4
cliff RMSE was unchanged/slightly worse and JAK2 overall RMSE degraded.  The
route was stopped before full-suite docking rather than selected on test.

Subsequent train-only falsification tests did not justify another official-test
run. A PDBBind-only PSICHIC interaction view failed on four difficult targets,
and its released checkpoints require explicit overlap gates (the human
multitask data contain nine official-test target--ligand pairs; PDBBind contains
two). A single PSD operator-action SVR worsened both metrics. A
KerRead-inspired environment-distribution kernel was directionally positive on
all 30 validation sets, but only by `0.14%` overall and `0.10%` on cliffs.
Cross-fitted affine and hard-example calibration were likewise rejected.

Three train-only follow-ups were rejected.  Exact-MMP bit salience degraded to
`0.64210 / 0.73309`, showing that edit effects cannot be encoded as global bit
importance.  OOF affine de-shrinkage produced `0.63969 / 0.73209`, showing
that regression-to-the-mean is not the main cliff bottleneck.  A direct
target-conditioned antisymmetric MMP model had 85.98% exact-core test coverage
but failed (`0.89222 / 0.89063`): identical 2D edits do not have transferable
effects without the binding microenvironment.

The released MTPNet final epoch (not its test-selected best checkpoint) was
also rescored under target-macro metrics and obtained `0.72874 / 0.82079 / 0.67183`.
This rules out generic residue-token cross-attention as a shortcut;
the next protein-conditioned model needs explicit pocket/contact localization.

Two fixed target-conditioned representation tests were also completed without
selection: ChemBERTa-MTR+ESM gives `0.67110 / 0.73883 / 0.63602`, and replacing
ChemBERTa by official Uni-Mol v1 gives `0.67898 / 0.75558 / 0.63664`.  Both are
rejected.  Their low training loss and weak test result show that endpoint
foundation embeddings do not replace explicit transferable edit semantics.

Before the no-ensemble constraint, the original bounded action ensemble
obtained macro RMSE `0.66351`, cliff RMSE `0.74788`, and non-cliff RMSE
`0.61452`.  The published GraphCliff reference is `0.665 / 0.757 / 0.619`.

The guarded v3 variant uses official SVM as the base model, applies action
residuals only when the fixed validation split improves by at least `0.002`
RMSE, and uses a single `0.8` residual shrinkage.  It obtains macro RMSE
`0.66346`, cliff RMSE `0.74621`, and non-cliff RMSE `0.61518`; the per-target
overall-RMSE comparison is 25 wins, 5 ties, and 0 losses against official SVM.
Because this guard is an ensemble/guarded residual protocol derived after
MoleculeACE exploration, it is retained only as historical evidence and is not
a compliant main claim.

On the untouched Dablander D2, factor Xa, and Mpro benchmark, calibrated MMP
transport beats the same SVM in all six fixed seed-42 folds on overall, cliff,
and non-cliff RMSE.  Six-fold macro RMSE is `0.68958 / 0.95837 / 0.62483`
versus SVM `0.69819 / 0.97872 / 0.63019`.

On ACNet Small, fixed-seed action classification also supports the core
hypothesis.  With the same ExtraTrees classifier and official random split,
endpoint features reach macro AUC `0.93906`, action-only features reach
`0.95168`, and endpoint+action features reach `0.95983` across 110 tasks.

On ACNet Mixed_Screened with target-held-out splitting, sparse fixed-budget
SGD logistic regression gives endpoint AUC/AP `0.51028 / 0.07147`, action-only
`0.58001 / 0.08682`, and endpoint+action `0.53657 / 0.07517`.  The action-only
view therefore improves AUC by `13.66%` and AP by `21.48%` relative to the
endpoint view on unseen targets.

## Current five-route train-only audit

The five physics/response hypotheses in `AGENTS.md` have now each received a
fixed, leakage-controlled implementation. Free-energy/cycle operators, explicit
local curvature, ligand-side interaction-energy proxies, and a Boltzmann
free-ligand conformer ensemble were rejected or produced only sub-percent
gains. The one robust signal is a target-response kernel built from ChEMBL
off-target and assay profiles after removing the benchmark target and every
biologically equivalent target.

After freezing one profile-interaction candidate on the original holdout, a
30-target, five-fold audit over official training pools changed overall/cliff/
non-cliff RMSE from `0.64400 / 0.74168 / 0.58549` to
`0.59278 / 0.67215 / 0.54455`. This is `7.95% / 9.38% / 6.99%`, with
`139/150`, `133/150`, and `127/150` paired fold wins. After freezing the
candidate, a separate official-test lockbox run (test labels never used)
reached `0.56877 / 0.63929 / 0.52614` versus official SVM
`0.67120 / 0.75113 / 0.62373`, i.e. `15.26% / 14.89% / 15.65%` relative gains.
This remains below the 30% target and is reported as a lockbox result, not a
claim of test-set selection. See `reports/five_route_audit.md`,
`outputs/moleculeace_profile_kernel_cv5_v1/aggregate.cv5.json`, and
`outputs/moleculeace_profile_kernel_test_v1/aggregate.test.json`.

## Frozen combined model

All retained positive components are now fit as one fixed SVR kernel:
10 parts collision-free chemistry, 8 parts multichannel ChEMBL response
profile, 4 parts local physical free-energy proxy, and 1 part chemistry ×
direct susceptibility interaction. Five-fold train-only CV gives
`0.59186 / 0.67161 / 0.54353` RMSE/cliff/non-cliff, the best fixed recipe so
far. Full-train artifacts for all 30 targets (SVR pickle, train kernel, labels,
SMILES, and recipe) are in `outputs/moleculeace_combined_model_v1/`.

## Method in one sentence

Instead of asking a smooth absolute regressor to resolve an activity cliff,
the action model learns the directed finite difference caused by an explicit
local molecular edit and transports potency from measured analog anchors.

See `reports/literature_landscape.md` for baseline selection and
`reports/method_design.md` for the hypothesis, equations, safeguards, rejected
iterations, and planned ablations.

## Current constraint boundary

The main method must be a single model with a fixed recipe.  Model ensembles,
validation-selected thresholds, shrinkage scans, and seed searches are retained
only as exploratory evidence and are excluded from compliant claims. Current
single-model variants have not reached the requested 10% overall and 20%
cliff-RMSE improvement. The strongest compliant recipe is now the fixed
combined positive-signal kernel, with 8.10% overall and 9.45% cliff gain in
30-target five-fold training-only CV. Its full-train artifacts are available,
but the combined recipe has not been promoted to a test-set claim; the earlier
profile-only lockbox remains the separately reported test result.

# Cliff-memory
