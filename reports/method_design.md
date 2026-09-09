# Method hypothesis: Action-Conditioned Cliff Transport

## Mechanistic diagnosis

Most molecular regressors impose an implicit smoothness prior: nearby molecular
representations should have nearby labels.  An activity cliff is exactly a
local violation of that prior.  Message passing can worsen the problem because
the unchanged scaffold dominates pooling while the few changed atoms are
smoothed into a nearly identical whole-molecule vector.

The changed atoms are not nuisance noise.  They are a **directed chemical
action** whose effect depends on the retained scaffold/local environment and
target.  Predicting two absolute endpoints independently wastes the strongest
supervision: their common core and assay offset cancel in the finite difference.

## Minimal compliant model

For molecule `m`, target context `t`, and a same-target training edit
`a -> q`, one scalar potential is learned:

```text
y_hat(m,t) = f_theta(m,t)
delta_hat(a -> q,t) = f_theta(q,t) - f_theta(a,t)
```

The absolute and finite-difference losses train the same parameters and the
same output.  This makes the action field antisymmetric and path-independent
by construction.  There is no separately trained fallback, vote, residual
blend, gate, or checkpoint selection in the compliant model.

The first learned representation is deliberately small:

- common ECFP bits represent retained context;
- source-only bits represent the removed action;
- destination-only bits represent the added action;
- hashed count-ECFP difference retains action direction and multiplicity;
- scalar similarity/edit-size features express reliability.

A shared action scorer is evaluated in both directions, making the predicted
delta antisymmetric by construction.  Later versions add triangle cycle loss,
which makes anchor-to-query reconstructions path-independent.

## Iteration record

1. **Endpoint action MLP (rejected).** It fit training-pair deltas almost
   perfectly but validation RMSE rose from 0.6730 to 0.6969 on CHEMBL204_Ki.
   Endpoint fingerprints allowed memorization instead of transferable actions.
2. **Explicit one-cut MMP transport.** A substitution is used only after it is
   observed on at least two *other* cores.  Across all 30 train-only validation
   folds, confidence blending changed macro RMSE 0.6742 → 0.6722 and cliff RMSE
   0.7527 → 0.7494, but lost cliff RMSE on 14 targets.
3. **Absolute core + substituent potential (rejected).** Although exactly
   antisymmetric and cycle-closed, sparse disconnected fragment identities made
   absolute prediction fail (CHEMBL204_Ki RMSE 0.7697).
4. **Calibrated MMP transport (current).** A stable hash chooses one train-only
   calibration fold.  Its analytic least-squares coefficient shrinks the MMP
   correction to the range [0, 1]; no grid, seed selection, or test label is
   involved.  The provisional validation macro is 0.6679 overall / 0.7412 cliff.
5. **Two-view anchored action ensemble.** A bounded two-variable least-squares
   solve on the fixed validation split combines (i) the RBF→Tanimoto kernel
   shift and (ii) the confidence-weighted MMP action correction.  Frozen test
   macro RMSE is 0.66351 overall, 0.74788 cliff, and 0.61452 non-cliff.  This
   clears the published GraphCliff macro values, but does not dominate SVM on
   every individual target; that stronger condition remains open.
6. **Guarded residual transport (exploratory v3).** The same two residuals are
   anchored on the official SVM baseline.  A residual is used only when the
   fixed validation split improves by at least 0.002 RMSE, then a single 0.8
   shrinkage is applied to all active targets.  On MoleculeACE this gives
   macro RMSE 0.66346 overall, 0.74621 cliff, and 0.61518 non-cliff, with
   25 wins, 5 ties, and 0 overall-RMSE losses versus official SVM.  Because
   this guard was derived after MoleculeACE exploration, it is a candidate
   protocol rather than fresh-lockbox proof.
7. **Single-model action potential variants (rejected).** After the no-ensemble
   constraint, three single-model variants were tested: closed-form
   action-regularized ridge, a neural potential with pair-delta consistency,
   and a BindingDB-pretrained single-anchor action-delta transport model.  They
   all underperformed SVM on the MoleculeACE failure targets, showing that
   ligand-only signed action transfer is not enough.
8. **KerRead/ACRead-inspired prototype transport (rejected as ligand-only).**
   KerRead suggests reading graphs through kernel centers, while ACRead uses
   centrality-conditioned attention to focus on structurally important nodes.
   A fixed 32-prototype action-kernel transport model was implemented as a
   single model, but it still failed on the key MoleculeACE targets.  The useful
   lesson is not to reuse it directly; it points to a target-conditioned action
   readout where the protein/assay context controls which edit prototypes are
   relevant.
9. **Single multitarget action potential (current compliant signal).** A shared
   ligand encoder and target embedding table were trained once across all 30
   MoleculeACE training splits, with same-target analog pair-delta consistency.
   The global-scale v1 model reaches macro RMSE 0.65593 overall / 0.73040 cliff
   / 0.61349 non-cliff versus official SVM 0.67120 / 0.75113 / 0.62373.  A
   fixed action-balanced objective that upweights train-only large-delta analog
   edges improves cliff RMSE further to 0.72769 and wins cliff RMSE on 19/30
   targets, but overall RMSE weakens to 0.65817.  Per-target activity scaling
   reduced training loss but did not improve macro cliff generalization, so it
   is not the main route.
10. **Protein-conditioned frozen-representation potential (rejected).** A
    single FiLM potential was trained once with frozen ChemBERTa-MTR ligand
    embeddings, frozen ESM2-T30 target embeddings, ECFP, receptor class, and
    assay type.  It used 777,019 train-only directed analog edges and no
    validation or checkpoint selection.  Despite near-zero training loss, test
    macro RMSE was 0.67110 overall / 0.73883 cliff / 0.63602 non-cliff.  This
    rejects the hypothesis that adding 1D ligand and protein foundation
    embeddings alone fixes the cliff curvature.
11. **Equal-weight multiscale kernel control.** One precomputed Tanimoto kernel
    averages radius-1/2/3 ECFP similarities and uses the same fixed SVR
    constants for all targets.  It reaches 0.65686 overall / 0.74370 cliff /
    0.60649 non-cliff and beats official SVM overall on 26/30 targets.  Wider
    context improves ordinary generalization substantially more than cliff
    generalization, motivating the frozen 3D Uni-Mol test rather than another
    kernel-weight or loss-weight scan.
12. **Frozen Uni-Mol 3D substitution (rejected).** The chemical encoder input
    was replaced by official Uni-Mol v1 representations for all 35,633 unique
    benchmark molecules, while every optimizer, architecture, seed, epoch, and
    action edge stayed fixed.  Macro RMSE degraded to 0.67898 overall / 0.75558
    cliff / 0.63664 non-cliff.  A deterministic free-ligand RDKit conformer is
    not the bioactive pose, so generic 3D pretraining does not supply the
    missing target-specific cliff geometry.
13. **Local-chemistry kernel (current strongest compliant overall).** One
    equal-weight PSD kernel combines chiral Morgan radius 1/2/3,
    pharmacophore-Morgan, and chiral atom-pair similarities.  With the same
    fixed SVR constants it reaches 0.63990 overall / 0.73089 cliff / 0.58893
    non-cliff.  This validates chemical-view coverage but still misses the
    requested cliff gain by a wide margin.
14. **Exact-MMP salience metric (rejected).** Train-only squared MMP effects
    produced closed-form fingerprint weights.  Performance degraded to 0.64210
    / 0.73309, falsifying target-wide bit salience: edit effects are conditional
    on their local binding context.
15. **Conditional antisymmetric MMP retrieval (rejected).** A single neural
    action function trained on 364,555 exact MMP pairs and used one nearest
    anchor at inference.  Despite 85.98% exact retained-core coverage and low
    training loss, it obtained 0.89222 / 0.89063 / 0.89780.  High coverage plus
    poor transfer identifies missing protein-pocket/conformational context, not
    a retrieval shortage.
16. **Analytic OOF de-shrinkage (rejected).** A fixed five-fold train-only OLS
    calibration on the local-chemistry kernel learned a mean slope of 1.004 and
    produced 0.63969 / 0.73209 / 0.58831.  Cliff error is therefore not mainly
    ordinary regression-to-the-mean bias.
17. **MTPNet released final checkpoint audit (rejected as evidence).** To avoid
    its test-as-validation checkpoint selection, the unselected epoch-50 model
    was evaluated directly.  It gives 0.72874 overall / 0.82079 cliff / 0.67183
    non-cliff macro RMSE.  Residue-token cross-attention alone is therefore not
    evidence for the required gain; pocket geometry/contact localization is the
    remaining mechanistic distinction.
18. **Experimental holo-pocket audit.** A fixed, activity-label-free PDBe/RCSB
    rule maps all 29 unique UniProt targets to an experimental ligand-bound
    structure and extracts residues within 6 Å. Alternate locations and
    modified polymer residues are excluded, and at least eight surrounding
    residues are required. Coverage is 29/29; this is a reusable target asset,
    not evidence of predictive improvement by itself.
19. **Pocket-gated pharmacophore kernel (rejected).** Six ligand pharmacophore
    edit views were weighted by complementary holo-pocket residue contacts and
    combined with the five local chemistry views as one PSD kernel. It reached
    0.64468 / 0.73357 / 0.59578, worse than the ligand-only local kernel. A
    static reference-pocket composition is too coarse to resolve directional
    substituent effects, so a larger cross-attention model was not justified.
20. **Fixed CliffLoss transfer (rejected).** The May 2026 CliffLoss severity
    score was applied train-only with its published fixed lambda 0.1. The
    validation-feedback controller was omitted because it adaptively tunes the
    loss weight. Result 0.64016 / 0.73094 / 0.58932 was effectively unchanged.
21. **Common/uncommon atom potential (rejected on internal validation).** A
    fixed four-layer GINE assigns scalar atom contributions, cancels aligned
    common-scaffold contributions, and matches train-only uncommon action
    deltas. Four hard-target validation RMSE values remained about 1.01--1.06,
    even with up to 755 aligned cliff pairs. Strict additive atom energy and
    invariant common-scaffold attribution are too restrictive; official test
    was not evaluated.
22. **Collision-free binary+count kernel (current strongest compliant).** A
    single SVR preserves occurrence counts and replaces 1024-bit hashes by an
    exact sparse training vocabulary across the same five chemical views. Both
    changes passed the fixed internal validation comparison before one official
    test run. It reaches 0.62391 overall / 0.71783 cliff / 0.57211 non-cliff,
    corresponding to 7.04% / 4.43% / 8.28% gains over official SVM. This is the
    strongest honest result, but it still fails the 10%/20% gate.
23. **Pose-action interaction kernel (rejected after staged validation).** A
    fully deterministic GNINA/Vina protocol (`seed=42`, `exhaustiveness=1`, one
    pose, CNN scoring disabled) docked every official-train molecule for CLK4
    and JAK2 into activity-label-free holo pockets. A single SVR added residue
    contact and Vina-affinity kernels to the ten exact 2D views. Internal
    validation improved both metrics on both targets, including CLK4 cliff RMSE
    0.89811 to 0.84632. The frozen official-test confirmation did not transfer:
    CLK4 cliff RMSE changed 1.11730 to 1.11799, while JAK2 overall RMSE changed
    0.56753 to 0.57920. The route was therefore stopped before a 30-target run;
    low-exhaustiveness single-pose docking does not supply a stable cliff signal.
24. **Frozen PSICHIC interaction view (rejected on internal validation).** The
    released human multitask checkpoint is inadmissible: canonical-SMILES and
    exact/contained-target-sequence auditing found nine official-test
    target--ligand pairs in its labeled human interaction data. Even the
    PDBBind-2020 checkpoint contains two official-test target--ligand pairs, so
    it cannot be used unconditionally across all 30 tasks. On four
    preregistered difficult, uncontaminated targets, adding one frozen 400-D
    interaction RBF view to ten exact chemical views worsened macro RMSE from
    0.71830/0.86831 to 0.72559/0.88355 (overall/cliff). No test labels were
    evaluated and the route was stopped.
25. **Operator action kernel (rejected).** A single mathematically PSD RKHS
    potential was trained on point evaluations and train-only finite-difference
    functionals using the operator kernel `A K A^T`. Close-edge mass was
    analytically normalized to point mass, with no lambda search. On the four
    hard targets it worsened overall/cliff validation RMSE from
    0.72026/0.86955 to 0.74291/0.88823. Reasserting differences already implied
    by absolute labels does not create transferable cliff information.
26. **Environment-distribution kernel (weak positive, not promoted).** Inspired
    by KerRead, exact atom-environment count measures were L1-normalized and
    compared with a train-median Laplacian kernel. Adding it as the eleventh
    equal atomic view improved all-30 validation macro RMSE from
    0.64025/0.72668/0.59375 to 0.63936/0.72597/0.59253, with 19/30 overall and
    cliff wins. The relative gains (0.14% overall, 0.10% cliff) are too small
    to justify another official-test evaluation.
27. **Cross-fitted calibration and hardness curriculum (rejected).** A fixed
    five-fold train-only OOF affine calibration slightly improved cliff RMSE on
    four hard targets (0.86955 to 0.86825) but worsened overall RMSE. A
    scale-free residual-rank curriculum worsened both to 0.72508/0.87857. This
    also avoided the supplied cliff flags, which may encode relationships
    computed across the official split. The result rules out further scalar
    calibration or reweighting searches.
28. **Target-response bioactivity profile kernel (robust positive, not yet
    promoted to official test).** Molecule-specific ChEMBL measurements on the
    benchmark target and every biologically equivalent target were removed.
    The remaining off-target and assay-level profiles were combined with exact
    local chemistry in one precomputed-kernel SVR. The candidate was frozen
    after the original seed-42 holdout, then audited with five-fold CV over all
    official training pools. Macro RMSE changed 0.64400 to 0.59278 and cliff
    RMSE 0.74168 to 0.67215, gains of 7.95% and 9.38%. It won 139/150 paired
    folds overall and 133/150 on cliffs, and 29/30 and 28/30 target means.
    Official test was not read. This is the only substantial signal among the
    five physics/response routes, but it does not pass the 30% requirement.
29. **Discrete local-curvature response (rejected).** A chemically weighted
    local response surface used 17 fixed steric, charge, solvation, and
    flexibility coordinates. Degree one and diagonal degree two were compared
    once on all 30 train-only validation splits. The baseline 0.64025/0.72668
    became 0.72826/0.80559 for the linear model and 0.73142/0.79908 for the
    curvature model. Explicit second differences therefore do not compensate
    for missing binding-state coordinates.
30. **Boltzmann conformer ensemble (rejected on the preregistered four-target
    pilot).** Eight deterministic ETKDGv3 conformers were MMFF94s/UFF optimized
    and aggregated at 298.15 K using energy-weighted means, variances, entropy,
    and effective conformer count. The ensemble-only kernel obtained
    1.09071/1.24445; adding it as one view to the exact 2D kernel degraded
    0.72026/0.86955 to 0.72887/0.88816. Along with the frozen Uni-Mol and staged
    docking audits, this rejects target-free gas-phase conformer populations as
    a route to cliff generalization.
31. **Direct assay-susceptibility score (rejected as an endpoint, retained only
    inside the richer profile kernel).** Five-fold cross-fitted univariate
    assay-to-benchmark response slopes were collapsed into one response score.
    Its standalone kernel failed at 1.13966/1.19981; a fixed 10:1 chemical-plus-
    score kernel improved the baseline only to 0.63073/0.71684. The full sparse
    response geometry and chemical-profile interaction are essential; a scalar
    susceptibility cannot explain the signal.

## External lockbox

After freezing calibrated MMP transport on MoleculeACE, it was evaluated on the
three datasets from Dablander et al. using their documented two-fold, seed-42
protocol.  No model setting was changed.  It beat the identical ECFP-SVM in all
six folds and in all three reported subsets (overall/cliff/non-cliff).  Macro
RMSE changed as follows:

| Method | Overall | Cliff | Non-cliff |
|---|---:|---:|---:|
| ECFP-SVM | 0.69819 | 0.97872 | 0.63019 |
| Calibrated MMP action | **0.68958** | **0.95837** | **0.62483** |

This external result supports transfer of the mechanism, while the remaining
MoleculeACE per-target losses delimit the claim honestly.

## ACNet action classification

ACNet Small was used as a fresh, fixed-seed classification check of the same
bottom-level hypothesis: cliff labels should be more predictable from the edit
than from endpoint identity alone.  Using ACNet's seed-8 random split and one
fixed ExtraTrees classifier across all 110 tasks:

| Feature view | Macro AUC | Macro AP | Macro ACC |
|---|---:|---:|---:|
| Endpoint only | 0.93906 | 0.74360 | 0.93764 |
| Action only | 0.95168 | 0.80237 | 0.94861 |
| Endpoint + action | **0.95983** | **0.80084** | **0.95266** |

Per-target wins are not universal because the Small test folds often contain
only 1-3 positive cliffs.  The larger ACNet Mixed target-held-out setting was
therefore added as a stronger sparse evaluation.

The sparse target-held-out implementation was then run on
MMP_AC_Mixed_Screened.  With a fixed-budget SGD logistic classifier:

| Feature view | AUC | AP | ACC |
|---|---:|---:|---:|
| Endpoint only | 0.51028 | 0.07147 | 0.67436 |
| Action only | **0.58001** | **0.08682** | 0.67808 |
| Endpoint + action | 0.53657 | 0.07517 | **0.70334** |

Relative to endpoint-only, action-only improves unseen-target AUC by 13.66%
and AP by 21.48%.  This reaches the desired 10%/20% scale for cliff
classification, but not yet for MoleculeACE cliff-RMSE regression.

## 20% cliff-RMSE gap

The current regression family cannot honestly claim a 20% cliff-RMSE gain.
For the strongest collision-free kernel, a 5,000-draw paired target/molecule
cluster bootstrap gives a 95% relative-gain interval of 5.54%--8.55% overall
and 1.77%--6.93% on cliffs.  The empirical probability of reaching either the
requested 10% overall or 20% cliff threshold is 0/5,000; the shortfall is not
plausibly explained by sampling noise in this evaluation.
An oracle that chooses the best prediction per test molecule among official
SVM, Tanimoto SVM, raw MMP, calibrated MMP, and guarded v3 still reaches only
0.68765 macro cliff RMSE versus official SVM 0.75113, an 8.45% relative gain.
This falsifies further tuning of the current ensemble family as a route to
20%.  The next method must use stronger action supervision or a representation
that resolves the local three-dimensional environment; a separately trained
cliff gate is excluded because it would recreate a guarded mixture.

The no-ensemble constraint tightens this conclusion.  Future candidates must be
one model with a fixed recipe: target/protein context, action prototype readout,
and a potential whose differences are trained on signed pair deltas.  Mixtures
of separately trained SVM/MMP/action predictors, validation thresholds,
shrinkage scans, and seed searches are excluded from the main method.

The first compliant multitarget action potential partially addresses this:
single model, fixed seed, no selection on test labels, and direct signed
pair-delta supervision.  Its best cliff gain is still only 3.12%, so the next
step cannot be another scalar weighting or normalization variant.  The missing
component is target-conditioned edit semantics: the model must know when an
edit is pharmacophorically meaningful for a given protein/assay rather than
only chemically local.

## Non-negotiable evaluation protocol

1. Keep MoleculeACE's official train/test split byte-for-byte unchanged.
2. Derive validation only from official training data with fixed seed 42.
3. Never choose a checkpoint, blend, threshold, or architecture on test labels.
4. Use one global configuration for every target; no per-target search.
5. Report RMSE, cliff RMSE, non-cliff RMSE, per-target predictions, and paired
   differences versus each baseline.
6. Separate transductive analog retrieval (train anchors only) from any
   genuinely target-held-out ACNet experiment.

## Required ablations

- absolute ECFP-SVM;
- nearest-neighbor label transport without an action model;
- action delta without core context;
- action delta with core context;
- remove antisymmetry;
- remove cycle consistency;
- learned anchor transport without the global fallback;
- fixed action model on ACNet target-held-out Mix.

## What would falsify the idea

The action hypothesis is not supported if gains disappear after removing exact
or near-duplicate transformations across splits, if improvement is confined to
one target, or if the same method improves global RMSE while degrading cliff
RMSE.  Those outcomes trigger redesign rather than seed or test-set tuning.
