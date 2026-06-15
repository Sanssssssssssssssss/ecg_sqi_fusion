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

## Stat-Fed Patchify Follow-up

I then tested the user's idea of feeding high-value waveform statistics into patchification itself.  This still keeps inference waveform-only: the model receives the ECG waveform, and all statistics are computed from that waveform inside the model/tokenizer.  It does not receive the 47-column teacher table as an input.

Two variants were added:

- `StatFedPatchTransformer`: each raw patch token gets local patch statistics such as mean/std/rms/ptp/diff/flatline/low-amplitude/zcr/slope, plus a global waveform-stat token.
- `MultiScaleStatPatchTransformer`: the same raw patch path plus extra statistic tokens at 50/100/250/500 sample windows, so attention can compare short QRS/detail windows and longer baseline/contact windows.

This worked architecturally, but not enough for BUT:

| candidate | synthetic test acc | original test acc | original all acc | original good/medium/bad | bad core | bad outlier stress |
|---|---:|---:|---:|---:|---:|---:|
| `statfed_patch_tx` | 0.9964 | 0.7755 | 0.8041 | 0.898 / 0.722 / 0.273 | 0.941 | 0.000 |
| `statfed_patch_medium_guard` | 0.9954 | 0.8057 calibrated | 0.8010 calibrated | 0.893 / 0.791 / 0.200 | 0.689 | 0.000 |
| `multiscale_statpatch_balanced` | 0.9944 | 0.7680 | 0.8409 calibrated | 0.916 / 0.691 / 0.287 | 0.992 | 0.000 |
| `multiscale_statpatch_medium_badguard` | 0.9944 | 0.8050 | 0.8356 | 0.904 / 0.772 / 0.290 | 1.000 | 0.000 |

The best thing this proved is not the headline accuracy.  It proved the model can recover more waveform-derived teacher axes when those statistics are embedded into tokens:

| feature | best stat-fed original corr | interpretation |
|---|---:|---|
| `pc1` | 0.974 | PCA morphology axis is learnable from waveform |
| `pca_margin` | 0.889 | margin is mostly recoverable |
| `non_qrs_diff_p95` | 0.949 | local-detail tail is strongly learnable |
| `sqi_bSQI` | 0.899 | beat-consistency proxy is learnable |
| `qrs_visibility` | 0.380 | improved but still weak |
| `knn_label_purity` | 0.351 | slightly improved, still far below tabular map quality |
| `boundary_confidence` | 0.162 | still not recovered |

So stat-fed patchify is the right architectural direction for local morphology, but it does not reconstruct the dataset-level neighborhood map.

## Bad Stress Shell Follow-up

The original bad-outlier-stress bucket was then compared directly against synthetic bad using the same `robust3` channels the Transformer consumes.  The gap is very large:

| feature | KS stress vs synthetic bad | stress median | synthetic bad median | interpretation |
|---|---:|---:|---:|---|
| `wf_diff_flat_015` | 1.000 | 0.3823 | 0.0104 | original stress has long flat/contact spans |
| `wf_z_flat_015` | 1.000 | 0.3082 | 0.0120 | same flatness in normalized waveform |
| `wf_diff_lowamp_050` | 1.000 | 0.6272 | 0.0408 | derivative channel is mostly low-amplitude |
| `wf_z_zcr` | 0.992 | 0.0320 | 0.2986 | far fewer zero crossings |
| `wf_baseline_ptp` | 0.975 | 4.3969 | 0.2776 | baseline channel has huge drift/span |

This explains why ordinary bad class weighting and noise augmentation failed: the missing bad stress is not simply noisier.  It is flatter/contact-like with a large baseline component.

I added a PTB-only `bad_stress_shell` augmentation in the waveform-input space.  It created a small improvement but not a solution:

| candidate | original test acc | bad core | bad outlier stress |
|---|---:|---:|---:|
| `multiscale_statpatch_stress_shell_badcal` | 0.7682 | 0.9412 | 0.0411 |
| `statfed_patch_stress_shell_badcal` | 0.8039 | 1.0000 | 0.0171 |

This is useful evidence: the shell direction is real because stress recall moved above zero, but forcing it through the current PTB bad samples hurts good/medium transfer and still does not approach the tabular bound.

## Fixed Waveform Ensemble Check

Finally, I averaged fixed waveform-only model probabilities to test whether the single models were complementary.  Weights were fixed by model role, not tuned on original BUT.

The best fixed ensemble reached only `0.8092` on original test, with bad outlier stress still `0.0000`.  The ensemble recovered bad core, but it did not recover the full original-test boundary:

| ensemble | original test acc | original good/medium/bad | bad core | bad outlier stress |
|---|---:|---:|---:|---:|
| `ens_medium_multiscale_equal_badcal` | 0.8092 | 0.899 / 0.784 / 0.290 | 1.000 | 0.000 |
| `ens_balanced_three_equal` | 0.7947 | 0.905 / 0.751 / 0.290 | 1.000 | 0.000 |
| `ens_all_profiles_equal_badcal` | 0.7954 | 0.897 / 0.759 / 0.290 | 1.000 | 0.000 |

This means the current waveform-only models make highly correlated original-domain mistakes.  A learned multi-expert head may still be useful, but simple averaging is not enough.

## Updated Decision

The strongest new result is methodological:

1. Feed computed waveform statistics into patch tokens.  This is worth keeping.
2. Use multi-scale statistic tokens for baseline/contact/detail context.  This improves original-all transfer and bad-core stability.
3. Do not expect class weights, stress-shell augmentation, or fixed ensembling alone to reach the 47-feature MLP.

The next serious architecture should be:

`Waveform Atlas Transformer with Learned Statistic Tokens`

- Tokenizer: raw patch tokens + local statistic tokens + multi-scale statistic tokens.
- Auxiliary heads: teacher feature recovery, beat/QRS visibility map, and bad-stress flat/contact map.
- Persistent memory: train-split synthetic atlas prototypes computed by boundary block and class.
- Loss: classify labels, recover teacher axes, and match prototype-neighborhood ordering against the persistent atlas.

The key missing ingredient is still the dataset-level atlas relation (`boundary_confidence` / `knn_label_purity`), not the ability to compute local SQI-like statistics from waveform.

## 2026-06-15 Continued Transformer Search

This report summarizes the continued waveform-only Transformer research after adding atlas-teacher, dual-bad, stress-bank, stress-token, high-stress, and waveform-stat-token variants. The classifier still receives waveform-derived inputs only; the 47 SQI/geometry columns remain training teacher targets or analysis labels, not inference inputs.

## What Changed

- Added teacher-atlas prototype distance features built from synthetic train geometry targets.
- Added a dual bad-vs-nonbad auxiliary head and bad-logit fusion for core/stress guardrails.
- Added waveform-derived stress bank features for long flat/contact spans, baseline drift, low derivative, low zero-crossing, and detail ratios.
- Tested stress as late fusion, stress as an attention token, stronger synthetic bad-stress augmentation, and a Transformer over waveform-derived statistic tokens.
- All selection remains synthetic/node diagnostic only. Original BUT buckets are report-only.

## Best Report-Only Original Test

| family | candidate | original_test_all_10s+_acc | original_test_all_10s+_good | original_test_all_10s+_medium | original_test_all_10s+_bad |
| --- | --- | --- | --- | --- | --- |
| dualbad_atlas | statfed_patch_dualbad_atlas | 0.813731 | 0.856593 | 0.826932 | 0.291971 |
| dualbad_atlas | statfed_patch_dualbad_atlas_badcal | 0.812434 | 0.856319 | 0.823995 | 0.299270 |
| highstress_atlas | stressbank_multiscale_highstress | 0.796980 | 0.907418 | 0.759602 | 0.221411 |
| highstress_atlas | stressbank_multiscale_highstress_badcal | 0.793677 | 0.907418 | 0.746046 | 0.299270 |
| dualbad_atlas | multiscale_statpatch_dualbad_atlas | 0.792025 | 0.854945 | 0.785133 | 0.309002 |
| stressbank_atlas | stressbank_multiscale_atlas | 0.791200 | 0.841484 | 0.795978 | 0.294404 |
| stressbank_atlas | stressbank_multiscale_atlas_badcal | 0.790846 | 0.840385 | 0.793945 | 0.318735 |
| dualbad_atlas | multiscale_statpatch_dualbad_atlas_badcal | 0.789666 | 0.854396 | 0.779937 | 0.321168 |

## Best Report-Only Original All

| family | candidate | original_all_10s+_acc | original_all_10s+_good | original_all_10s+_medium | original_all_10s+_bad |
| --- | --- | --- | --- | --- | --- |
| stressbank_atlas | stressbank_statfed_atlas | 0.846341 | 0.819926 | 0.845879 | 0.932450 |
| stressbank_atlas | stressbank_statfed_atlas_badcal | 0.845400 | 0.819163 | 0.843338 | 0.934153 |
| stresstoken_atlas | stresstoken_statfed_atlas_badcal | 0.826344 | 0.802265 | 0.821321 | 0.914096 |
| stresstoken_atlas | stresstoken_multiscale_atlas_badcal | 0.814692 | 0.746465 | 0.865262 | 0.933018 |
| stresstoken_atlas | stresstoken_multiscale_atlas | 0.814328 | 0.746465 | 0.865450 | 0.930369 |
| statbank_token | statbank_token_balanced | 0.811142 | 0.751159 | 0.847384 | 0.931693 |
| statbank_token | statbank_token_badguard | 0.811112 | 0.765358 | 0.824991 | 0.930747 |
| statbank_token | statbank_token_balanced_badcal | 0.810475 | 0.750983 | 0.843432 | 0.936045 |

## Best Bad-Outlier Stress Slice

| family | candidate | bad_outlier_stress_acc | bad_outlier_stress_good | bad_outlier_stress_medium | bad_outlier_stress_bad |
| --- | --- | --- | --- | --- | --- |
| dualbad_atlas | multiscale_statpatch_dualbad_atlas_badcal | 0.044521 | 0.000000 | 0.000000 | 0.044521 |
| stressbank_atlas | stressbank_multiscale_atlas_badcal | 0.041096 | 0.000000 | 0.000000 | 0.041096 |
| statbank_token | statbank_token_badguard_badcal | 0.037671 | 0.000000 | 0.000000 | 0.037671 |
| dualbad_atlas | multiscale_statpatch_dualbad_atlas | 0.027397 | 0.000000 | 0.000000 | 0.027397 |
| highstress_atlas | stressbank_statfed_highstress_badcal | 0.023973 | 0.000000 | 0.000000 | 0.023973 |
| highstress_atlas | stressbank_multiscale_highstress_badcal | 0.020548 | 0.000000 | 0.000000 | 0.020548 |
| stressbank_atlas | stressbank_statfed_atlas_badcal | 0.017123 | 0.000000 | 0.000000 | 0.017123 |
| dualbad_atlas | statfed_patch_dualbad_atlas_badcal | 0.013699 | 0.000000 | 0.000000 | 0.013699 |

## Readout

- The best original_test_all_10s+ in this round is still only about 0.814 from statfed_patch_dualbad_atlas, far below the 47-feature tabular upper bound of 0.9635.
- Stress-bank variants improve original_all_10s+ up to about 0.846, but do not improve original_test_all_10s+ and still leave bad_outlier_stress near zero.
- Stress-token variants are unstable: attention over an explicit stress token can damage bad core or collapse bad stress rather than solve it.
- High-stress augmentation does not solve transfer. Larger bad stress shells pull decision boundaries but do not make original bad outliers reliably bad.
- Statbank-token Transformer confirms that waveform-computable statistics are not enough to match the reduced-feature or 47-feature tabular models; the missing signal is mainly dataset-level atlas relation such as boundary confidence / KNN purity, not only local morphology.

## Current Best Interpretation

The Transformer can learn local waveform morphology and synthetic node labels extremely well, but it still fails to infer the atlas-like neighborhood variables that the tabular model receives directly. The next serious model direction should therefore make the waveform encoder learn a retrieval/prototype relation in embedding space, not just add more local stats or stronger bad class weights.

Recommended next experiments:

1. Train a waveform embedding with explicit synthetic-neighbor retrieval loss: classify each sample by distances to frozen synthetic train prototypes in learned embedding space, then distill boundary_confidence and knn_label_purity from those distances.
2. Use two-stage contrastive pretraining on synthetic blocks before classification: block identity + class label + neighbor-purity ordering, then fine-tune the classifier head.
3. Treat original bad_outlier_stress as a separate domain-stress problem. More bad weighting alone hurts good/medium and does not recover stress bad.
4. Keep 47-feature MLP as upper bound and reduced-feature MLP as explainability bound; do not claim waveform-only has matched it yet.

References used for architecture direction: PatchTST patch tokens, MTECG masked ECG Transformer, supervised contrastive learning, and prototypical networks.
