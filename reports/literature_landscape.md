# Lightweight activity-cliff baselines

This is the frozen baseline shortlist for the first study.  “Lightweight” here
means that a method can be trained independently on one MoleculeACE target on a
single workstation GPU or CPU, without foundation-model pretraining.

| Method | Year | Why it is in the main comparison | Local asset |
|---|---:|---|---|
| ECFP + SVM/RF/GBM | 2022 | The strongest and most reproducible classical MoleculeACE references; SVM is the primary floor. | `third_party/MoleculeACE` |
| ACNet ECFP + MLP | 2023 | Lightweight pair-classification baseline and the source of the 190-target AC benchmark. | `third_party/ACNet`, `data/acnet` |
| GraphCliff | 2026 | A 3-layer, 6.0M-parameter short/long-range gated GNN designed specifically to avoid smoothing away local cliff edits. | `third_party/GraphCliff` |
| ACANet / ACA loss | 2026 | Recent activity-cliff-aware triplet/metric-learning objective that can wrap a small graph model. | `third_party/ACANet` |
| ACES-GNN | 2025 | Common-scaffold attribution suppression plus edit-region attribution ranking; released workflow also supports tuned weights/ensembles. | `third_party/XACs` |
| MTPNet | 2025 | Residue-level protein-to-ligand cross-attention is useful inspiration, but the released default regression script uses test as validation. | `third_party/MTPNet` |
| Curriculum-aware training | 2025 | Molecule-similarity graph with node curriculum and edge-level cliff discrimination. | paper/code reference |
| SCAGE | 2025 | Functional-group and multiscale-conformation pretraining; useful representation evidence, not a 20% cliff solution. | source tables under `data/` |
| CliffLoss | 2026 | Train-only local severity reweighting; fixed-weight ablation is compatible, adaptive validation feedback is not. | arXiv source audited locally |

The benchmark foundation is [MoleculeACE](https://doi.org/10.1021/acs.jcim.2c01073),
with its [official code and 30 splits](https://github.com/molML/MoleculeACE).
The pair-classification complement is [ACNet](https://arxiv.org/abs/2302.07541)
and its [official repository](https://github.com/DrugAI/ACNet/).  The two most
recent cliff-specific references are [GraphCliff](https://arxiv.org/abs/2511.03170)
([code](https://github.com/dmis-lab/GraphCliff)) and
[ACANet](https://www.nature.com/articles/s41467-026-75713-2)
([code](https://github.com/shenwanxiang/ACANet)).

Methods retained as secondary literature rather than first-line baselines are
[SemiMol](https://www.ijcai.org/proceedings/2024/0672.pdf),
[MaskMol](https://arxiv.org/abs/2409.12926), and
[MolMCL](https://www.nature.com/articles/s41467-024-55082-4).  They require
additional unlabeled corpora, image masking, or substantial pretraining, so a
comparison would conflate the cliff objective with extra-data/compute effects.

## Reproduction status

- The official MoleculeACE ECFP-SVM result is reproduced over all 30 targets:
  macro RMSE 0.671197, cliff RMSE 0.751128, non-cliff RMSE 0.623734.
- GraphCliff reports 0.665 / 0.757 / 0.619 for those three metrics.  A clean
  local CHEMBL204_Ki run reached 0.688768 overall and 0.815782 on cliffs, using
  the documented seed-42 train/validation protocol and 100-epoch ceiling.
- ACNet raw files were downloaded from the authors' Drive links and regenerated
  into all official size regimes; the complete mixed set contains 402,079
  labeled pairs across 190 targets.
- The Dablander D2, factor Xa, and SARS-CoV-2 Mpro tables are included from the
  authors' [experiment repository](https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments).

These numbers are reference points, not claims of a new winner.  A proposed
method must beat both the 30-target macro metrics and disclose per-target losses.

## 2025--2026 claim audit

- SCAGE Supplementary Tables S4/S5 give macro `0.67207` overall and `0.73177`
  cliff RMSE.  Against the official ECFP-SVM reproduced here, that is -0.13%
  overall and only +2.58% cliff, despite ten-seed averages and an enumerated
  hyperparameter search.
- ACANet reports `0.646 / 0.711` versus its `0.671 / 0.742` SVM.  The final
  protocol linearly searches cliff thresholds and loss weight, selects nested-CV
  checkpoints, and averages five submodels.  Even so, the reported gains are
  about +3.7% overall and +4.2% cliff, not +10%/+20%.
- MTPNet reports a large pooled-RMSE improvement, but `main.py` assigns official
  test rows to both validation and test and selects an epoch on that validation
  set.  The released metric pools targets instead of computing MoleculeACE
  target-macro/cliff metrics, so the headline is not leakage-controlled.  As an
  additional audit, its unselected final epoch-50 checkpoint was rescored here:
  `0.72874` overall / `0.82079` cliff / `0.67183` non-cliff macro RMSE.
- ACES-GNN supplies the most relevant mechanistic lesson: common-scaffold
  contributions should cancel and potency-difference attribution should lie on
  the uncommon edit.  Its tuned weights, repeated folds, and ensemble path are
  excluded by the present protocol.
- CliffLoss (arXiv:2605.17265) defines severity from local Tanimoto similarity
  times activity difference and reports up to 9.7% overall MAE improvement and
  30% compression of the cliff-to-smooth error gap on different benchmarks.
  Its full method adapts a loss weight from validation severity, so only the
  published fixed `lambda=0.1` component was transferred here; it did not
  improve MoleculeACE (`0.64016 / 0.73094`).
- A 2026 leakage-controlled kinase analysis reports that 97% of cliff-forming
  transformations occur on one kinase and median cross-kinase agreement is
  -0.058.  This agrees with the failed cross-target MMP retrieval experiment:
  transformation recurrence alone is not a defensible route to the 20% gate.
