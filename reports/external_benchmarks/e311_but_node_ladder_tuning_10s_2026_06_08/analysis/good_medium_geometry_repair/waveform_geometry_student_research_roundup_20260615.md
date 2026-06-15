# Waveform Geometry Student Research Roundup - 2026-06-15

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
