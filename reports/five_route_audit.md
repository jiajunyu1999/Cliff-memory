# Five-route train-only audit

All comparisons below use only MoleculeACE official training pools. The
official test set is untouched unless an older result is explicitly described
as historical. No result below was selected by changing a random seed.

| AGENTS.md route | Fixed implementation | Evidence | Decision |
|---|---|---|---|
| 1. Free-energy perturbation / cycles | Integrable action potentials, operator-action RKHS, and four fixed ligand free-energy proxy channels | Operator action worsened the four-target holdout from 0.72026/0.86955 to 0.74291/0.88823. The 30-target physical proxy kernel was 0.89077/0.94082; its best local product gave only 0.63789/0.72438 versus 0.64025/0.72668. | No scalable positive action/cycle signal; do not tune loss weights. |
| 2. Local curvature | Chemically local degree-1 response versus diagonal degree-2 response in 17 physical coordinates | 30-target overall/cliff RMSE: baseline 0.64025/0.72668, linear 0.72826/0.80559, quadratic 0.73142/0.79908. | Rejected; curvature in ligand-only coordinates is the wrong geometry. |
| 3. Steric/electrostatic interaction balance | Desolvation, charge, flexibility/strain, and size/dispersion RBF views, including chemistry-local products | Best 30-target result 0.63789/0.72438, only 0.37%/0.32% better than the strong 2D kernel. | Weak positive below promotion threshold. |
| 4. Boltzmann conformer ensemble | Eight ETKDGv3 conformers, MMFF94s/UFF energies, 298.15 K weighted shape mean/variance/entropy | Four-target baseline 0.72026/0.86955; combined model 0.72887/0.88816. | Rejected; target-free conformers do not approximate the bound ensemble. |
| 5. Susceptibility field | Leakage-filtered ChEMBL off-target/assay response profiles plus local chemistry and their product in one SVR | Frozen-candidate 30-target five-fold CV: 0.64400/0.74168 to 0.59278/0.67215, gains 7.95%/9.38%; 139/150 overall and 133/150 cliff fold wins. | Robust positive; deepest current route, but fails the 30% gate. |

The route-5 gain correlates with assay-transfer validation coverage (Spearman
approximately 0.48 overall and 0.45 on cliffs). CHEMBL4203 reaches 27.15%
overall and 39.37% cliff improvement, whereas sparse-profile CHEMBL4616 and
CHEMBL264 are flat or slightly negative. A direct scalar susceptibility score
does not solve this coverage problem: alone it gives 1.13966/1.19981 and when
added to chemistry gives only 0.63073/0.71684. The transferable signal is in
the multichannel response geometry, not a single fitted slope.

## Official-test lockbox

The route-5 candidate was frozen before reading official-test labels. Test-pool
ChEMBL profiles were fetched independently, and exact benchmark targets plus
biologically equivalent targets were removed. Across all 30 official targets
(9,802 test rows; 3,790 cliffs), macro RMSE/cliff/non-cliff is
`0.56877 / 0.63929 / 0.52614`; official SVM is `0.67120 / 0.75113 / 0.62373`.
Relative reductions are `15.26% / 14.89% / 15.65%`, still below the requested
30% gate. Full audit: `outputs/moleculeace_profile_kernel_test_v1/aggregate.test.json`.

## Combined trainable artifact

The strongest fixed recipe combines every component that survived the route
audit in one precomputed-kernel SVR: chemistry:profile:local-physics:
chemistry×susceptibility weights `10:8:4:1`, with `C=10` and epsilon `0.1`.
Five-fold CV on official training pools yields `0.59186 / 0.67161 / 0.54353`
(overall/cliff/non-cliff), relative gains `8.10% / 9.45% / 7.17%` over the
collision-free baseline. The model was then refit on all official training
rows; per-target `.svr.pkl`, `.train_kernel.npz`, and `recipe.json` artifacts
are stored under `outputs/moleculeace_combined_model_v1/`.

The next defensible experiment must improve target-conditioned response
coverage without querying benchmark-target labels: for example a pretrained
multi-assay latent response model with explicit biological-equivalence and
target--ligand overlap gates. It must be frozen on train-only cross-validation
and exceed the current profile kernel before any official-test evaluation.
