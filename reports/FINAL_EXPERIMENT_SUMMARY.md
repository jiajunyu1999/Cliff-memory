# ResponseKernel final experimental summary

## Scientific claim

Activity cliffs violate the smoothness assumption of chemistry-only similarity.
ResponseKernel therefore retains only two top-level sources of evidence:
collision-free local chemistry and a leakage-filtered external response field.
The benchmark target and biologically equivalent ChEMBL targets are removed;
all feature relevance, normalization, and kernel scales are fit inside the
training fold.

## Final model

The response field contains entity-level and assay-level sparse geometry,
fit-only dense relevance views, and chemistry-by-response interactions. Protein
sequence neighborhood, physical proxy, and explicit susceptibility branches
were removed because they did not show independent, significant improvement.

## Primary results

- MoleculeACE: 30 targets, three seeds, five folds per seed, 450 paired folds.
- Chemistry baseline: overall/cliff/non-cliff RMSE = 0.6428/0.7380/0.5854.
- ResponseKernel: overall/cliff/non-cliff RMSE = 0.5913/0.6693/0.5437.
- Relative reduction: 8.01%/9.32%/7.13%.
- Target-clustered 95% bootstrap interval for absolute RMSE reduction:
  [0.0372, 0.0678] overall and [0.0465, 0.0953] on cliffs.
- Target-averaged wins: 29/30 overall and 27/30 on cliffs; one-sided paired
  Wilcoxon p-values are below 2e-8.
- Official-test confirmation: 0.5687/0.6377/0.5266, corresponding to
  15.3%/15.1%/15.6% reductions versus the reproduced official ECFP-SVM.

## Retained-block evidence

Under the complete seed-42 five-fold ablation, the final overall/cliff RMSE is
0.5927/0.6727. Restrictions are uniformly worse: entity only 0.6032/0.6853;
assay only 0.6052/0.6884; raw profiles only 0.6072/0.6904; dense profiles only
0.6127/0.6959; without chemistry-response interactions 0.5959/0.6749. Removing
the interaction has target-level one-sided Wilcoxon p=2.5e-5 overall and
p=0.060 on cliffs; the overall effect is supported, while the cliff-specific
effect remains uncertain.

## Breadth and mechanism evidence

- Five external low-coverage GraphCliff assays: overall RMSE improves by 4.66%
  and cliff RMSE by 7.58%; non-cliff RMSE worsens by 3.31%, an explicit boundary.
- Full ACNet: 190 target tasks and 570 target/model evaluations. Adding edit
  features raises macro AUC/AP/accuracy from 0.8636/0.4069/0.8905 to
  0.8687/0.4301/0.9077.
- Total breadth: 225 target-level tasks across separately reported regression
  and matched-pair classification formulations.
- Official-test cases: median absolute-error reduction is 0.0506 over 9,802
  molecules and 0.0601 over 3,790 cliff molecules. Positive panels and
  zero-response-coverage boundary panels use declared selection rules.

## Reproducible outputs

- Final trained artifacts: `outputs/responsekernel_final_model_v1/`
- Repeated-CV statistics: `reports/results/responsekernel_final_evidence/`
- Internal ablation: `outputs/responsekernel_internal_ablation_cv5_seed42/`
- External assays: `outputs/responsekernel_final_external_cv5_seed42/`
- ACNet breadth: `outputs/acnet_all190_action_sgd_v1/`
- Case-study tables: `reports/results/case_studies_final/`
- Manuscript: `paper/activity_cliff_response.tex` and
  `paper/activity_cliff_response.pdf`
