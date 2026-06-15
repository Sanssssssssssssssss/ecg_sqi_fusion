# Waveform Geometry Architecture Diagnosis - 2026-06-15

## Question

Why can the 47-feature tabular model separate the BUT geometry, while waveform-only Transformer models still miss key regions?  Which architectural changes actually help the missing features?

The constraint remains strict: PTB synthetic waveform is the only training input.  BUT is report-only.  SQI/geometry features are teacher targets only and are not classifier inputs.

## Feature Failure Taxonomy

The previous waveform models did not fail uniformly.

| feature group | status | evidence | interpretation |
|---|---|---|---|
| global morphology axis | partly learned | `pc1`, `pca_margin`, `non_qrs_diff_p95` often reach original corr 0.84-0.94 | waveform contains enough information; stat/detail branches help |
| QRS visibility/prominence | weak | `qrs_visibility` original corr stays near 0.03-0.25 across models | current tokenization/pooling does not preserve a stable template/beat-level visibility signal |
| atlas/neighborhood geometry | weak | `boundary_confidence` and `knn_label_purity` stay around corr 0.1-0.3 | these are not pure per-window waveform features; they encode target manifold neighborhood structure |
| original bad outlier | still missing | `bad_outlier_stress` remains 0 for most waveform-only models | PTB synthetic bad stress does not yet match BUT 111001-like outlier morphology |

## Architecture Experiments

Three targeted architectures were added to `run_waveform_geometry_student.py`.

| candidate | intended fix | synthetic test acc | original test acc | original bad core | original bad outlier |
|---|---|---:|---:|---:|---:|
| `qrs_detail_token_tx` | preserve high-resolution local QRS/detail tokens | 0.8878 smoke | not run | not run | not run |
| `atlas_memory_tx` | add trainable prototype-distance memory for boundary/KNN-like structure | 0.9923 | 0.7782 | 1.0000 | 0.0000 |
| `qrs_atlas_memory_tx` | combine high-resolution QRS tokens with prototype memory | 0.9949 | 0.7956 | 0.9748 | 0.0000 |
| `teacher_atlas_student` | initialize atlas distances from PTB teacher prototypes | 0.9918 | 0.7963 | 0.0000 | 0.0000 |
| `qrs_teacher_atlas_student` | QRS-detail encoder plus PTB teacher atlas distances | 0.9933 | 0.8260 | 0.1513 | 0.0000 |
| `hybrid_atlas_student` | QRS-detail encoder plus random atlas memory plus teacher atlas distances | 0.9974 | 0.7798 | 1.0000 | 0.0000 |
| `neighbor_stattoken_student` | batch teacher-neighborhood KL on stat-token embedding | 0.9933 | 0.7779 | 0.9076 calibrated | 0.0000 |
| `neighbor_hybrid_atlas_student` | random+teacher atlas plus batch teacher-neighborhood KL | 0.9969 | 0.7939 | 1.0000 calibrated | 0.0000 |

## What Improved

`atlas_memory_tx` gives a real structural improvement on the synthetic/node geometry:

- synthetic test acc `0.9923`;
- synthetic good/medium/bad recall `0.981 / 0.999 / 0.979`;
- original bad core/near-boundary recall `1.0000`.

It also keeps strong original recovery for features that are close to waveform morphology:

- `pc1` corr `0.939`;
- `pca_margin` corr `0.886`;
- `non_qrs_diff_p95` corr `0.933`;
- `template_corr` corr `0.713`;
- `band_30_45` corr `0.769`.

## What Did Not Improve

The targeted architecture did **not** solve the hardest teacher features:

| model | `qrs_visibility` corr | `boundary_confidence` corr | `knn_label_purity` corr |
|---|---:|---:|---:|
| `stattoken_v2_badstress` | 0.120 | 0.185 | 0.312 |
| `morphology_token_tx` | 0.251 | 0.111 | 0.186 |
| `atlas_memory_tx` | 0.030 | 0.136 | 0.285 |
| `qrs_atlas_memory_tx` | 0.148 | -0.061 | 0.009 |

This is the key result: random trainable prototypes do not recreate the target atlas.  They help the model classify the synthetic manifold, but they do not give it the same neighborhood map that the tabular model receives as input.

## Teacher Atlas Follow-up

The teacher-initialized atlas experiments were designed to answer whether the missing map can be made explicit inside the waveform model while still keeping waveform-only inference.

The answer is: only partially.

- `qrs_teacher_atlas_student` raises original test acc to `0.8260`, but it still predicts almost all original bad as good/medium; bad core recall is only `0.1513` after bad calibration and bad outlier stress remains `0.0000`.
- `teacher_atlas_student` without the QRS-detail path keeps medium high but loses good and all bad core; original test acc is `0.7963`.
- `hybrid_atlas_student` recovers bad core (`1.0000`) by keeping random atlas memory, but it lowers medium recall to `0.7592` on original test and still has bad outlier stress `0.0000`.

This splits the failure into two different mechanisms:

| mechanism | best evidence | conclusion |
|---|---|---|
| local morphology/stat features | `pc1`, `pca_margin`, `non_qrs_diff_p95`, `sqi_bSQI` recover well from waveform | waveform has enough signal for many SQI-like axes |
| PTB teacher atlas distances | teacher-atlas synthetic acc is high, but original bad core can become 0 | PTB synthetic atlas is not enough to define BUT bad neighborhoods |
| random prototype memory | `atlas_memory_tx` and `hybrid_atlas_student` recover bad core | prototype memory can form a bad-core basin |
| global neighborhood purity | all waveform models have weak `boundary_confidence` / `knn_label_purity` recovery | MLP is strong because it directly receives target-neighborhood information |

Key original feature recovery correlations are saved in:

- `outputs/.../analysis/good_medium_geometry_repair/waveform_geometry_feature_recovery_key_compare.csv`
- `reports/.../analysis/good_medium_geometry_repair/waveform_geometry_feature_recovery_key_compare.csv`

The most important numbers:

| feature | best original corr | best candidate | interpretation |
|---|---:|---|---|
| `pc1` | 0.941 | `stattoken_v2_badstress` | learnable waveform morphology axis |
| `pca_margin` | 0.886 | `atlas_memory_tx` | mostly learnable from waveform/stat tokens |
| `non_qrs_diff_p95` | 0.933 | `atlas_memory_tx` | learnable local-detail/noise axis |
| `sqi_bSQI` | 0.895 | `atlas_memory_tx` | learnable beat-consistency proxy |
| `qrs_visibility` | 0.327 | `qrs_teacher_atlas_student` | still weak; scalar head is not enough |
| `boundary_confidence` | 0.226 | `teacher_atlas_student` | not recovered as a stable waveform scalar |
| `knn_label_purity` | 0.312 | `stattoken_v2_badstress` | not recovered; needs embedding-neighborhood training |

## Why the 47-feature MLP Wins

The MLP is not magically learning ECG morphology better than a Transformer.  It is given features that already contain global target geometry:

- PCA position and margin;
- target-region confidence;
- KNN label purity;
- boundary confidence.

Those are not simply local waveform descriptors.  They are properties of a sample relative to a fitted reference atlas.  A waveform-only model must reconstruct that atlas relation from waveform alone, which the current losses only approximate through per-sample scalar regression.

The architecture implication is precise: the next model should not only regress the teacher feature vector.  It should make the embedding geometry itself match the teacher-neighborhood geometry.

## Neighborhood Distillation Follow-up

I then added an explicit batch-level teacher-neighborhood KL loss.  For each training batch, the loss compares:

- the soft nearest-neighbor distribution in teacher core-feature space;
- the soft nearest-neighbor distribution in waveform embedding space.

This was intended to push the Transformer embedding toward the same geometry that makes the tabular model strong.

Result:

| candidate | synthetic test acc | original test acc | original good/medium/bad | bad core | bad outlier |
|---|---:|---:|---:|---:|---:|
| `neighbor_stattoken_student` | 0.9933 | 0.7779 calibrated | 0.762 / 0.839 / 0.263 | 0.9076 calibrated | 0.0000 |
| `neighbor_hybrid_atlas_student` | 0.9969 calibrated | 0.7939 calibrated | 0.912 / 0.744 / 0.290 | 1.0000 calibrated | 0.0000 |

The loss helps preserve synthetic/node quality, and it can recover bad core after calibration, but it does not improve the main original-test ceiling.  It also still does not recover the key target-neighborhood features:

| feature | best original corr after neighborhood KL | note |
|---|---:|---|
| `boundary_confidence` | 0.135 for `neighbor_hybrid_atlas_student` | still below the 0.226 teacher-atlas result and far too weak |
| `knn_label_purity` | 0.082 for `neighbor_hybrid_atlas_student` | worse than the 0.312 stat-token bad-stress baseline |
| `qrs_visibility` | 0.315 for `neighbor_hybrid_atlas_student` | slight improvement but still weak |

This rules out a simple batch-KL fix.  The missing object is not just pairwise smoothness inside a mini-batch.  It is a stable dataset-level memory of the teacher atlas.

## Architectural Conclusion

The next architecture should not be "larger Transformer" or "more class weight."  It should make the teacher map explicit in the representation learning objective.

Recommended next architecture:

`Neighborhood-Distilled Atlas Student`

- Build teacher clusters/prototypes from PTB synthetic train features for each boundary block and class.
- Compute teacher-neighborhood targets on train batches in the synthetic feature space.
- Add a memory-bank / prototype contrastive loss: waveform embedding should preserve teacher nearest-neighbor ordering against a persistent synthetic atlas, not only within mini-batches.
- Add a local-neighborhood distillation loss: predicted neighbor purity/boundary confidence should match teacher KNN computed against the persistent synthetic train atlas.
- Keep inference waveform-only: prototypes live inside the model checkpoint and are learned constants, not per-sample feature inputs.

Recommended QRS-specific fix:

- Replace scalar `qrs_visibility` regression with a beat/template auxiliary task.
- Predict local peak-mask / visibility map from waveform tokens, then pool it to the scalar teacher.
- This gives the model a beat-level path to learn QRS visibility rather than asking the final pooled embedding to infer it.

Recommended data/generator fix:

- Add a PTB-only `bad_111001_like_controlled_stress` block before more architecture sweeps.
- Target low `qrs_band_ratio`, low `qrs_visibility`, high `baseline_step`, high `band_30_45`, high `non_qrs_diff_p95`.
- Keep extreme bad as report-only stress until this controlled block improves without collapsing good/medium.

## Decision

Current best waveform-only research direction is still `stattoken_v2_badstress` for overall original acc (`0.8358`).  `atlas_memory_tx` and `hybrid_atlas_student` are the best evidence that prototype memory can recover bad core, but they do not solve good/medium transfer or bad outlier stress.

The next useful experiment is not another random architecture or larger Transformer.  It is a neighborhood-distilled atlas student plus beat-level QRS visibility auxiliary target, trained only on PTB synthetic data and evaluated on BUT report-only.  In parallel, the generator needs a controlled bad outlier block, because no current waveform-only architecture recovers bad outlier stress from the existing PTB synthetic boundary alone.
