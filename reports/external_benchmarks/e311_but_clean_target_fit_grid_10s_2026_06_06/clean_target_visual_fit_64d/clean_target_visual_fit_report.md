# CleanBUT-Core Visual Fit

This report projects PTB synthetic audit samples into a PCA fitted only on the CleanBUT train-target core.
It is CPU-only and uses signal-derived morphology/QRS features, not model predictions.

- Common feature count: `64`
- Feature mode: `full64`
- PCA explained variance: `0.655`, `0.147`

## Figures

- `figures/ptb_vs_cleanbut_pca_top_rules.png`: CleanBUT-Core background with top PTB synthetic variants overlaid.
- `figures/ptb_vs_cleanbut_best_rule.png`: single best current rule in the same PCA space.
- `figures/ptb_vs_cleanbut_centroid_shift.png`: class centroid shifts from CleanBUT to synthetic.
- `figures/clean_target_distance_bars.png`: class-wise distance bars from the no-training scan.

## Current Read

The closest no-training candidates are mostly `qrs_confound` and `visible_unusable` bad-subtype rules with `bw_badstrong` overlay and a protected medium guard. The 64-feature view is stricter: it checks morphology, SQI proxies, frequency bands, wavelet energy, and complexity features together.

## Top Variants

| rank | variant_id | full_feature_score | full_good_distance | full_medium_distance | full_bad_distance | n_common_features | mean_centroid_distance_common_pca |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5300 | 0.5763 | 0.3108 | 0.7480 | 64 | 7.9777 |
| 2 | mg_qrs_confound_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5301 | 0.5763 | 0.3112 | 0.7478 | 64 | 8.1704 |
| 3 | mg_visible_unusable_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5311 | 0.5763 | 0.3108 | 0.7508 | 64 | 7.7873 |
| 4 | mg_visible_unusable_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5314 | 0.5763 | 0.3128 | 0.7494 | 64 | 8.0770 |
| 5 | mg_qrs_confound_m_guard_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5320 | 0.5763 | 0.3112 | 0.7529 | 64 | 8.1425 |
| 6 | mg_visible_unusable_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5322 | 0.5763 | 0.3108 | 0.7537 | 64 | 7.9232 |
| 7 | mg_contact_lowamp_m_guard_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5332 | 0.5763 | 0.3112 | 0.7560 | 64 | 7.9998 |
| 8 | mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5336 | 0.5763 | 0.3128 | 0.7552 | 64 | 8.0252 |
| 9 | mg_balanced_or_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5340 | 0.5763 | 0.3112 | 0.7579 | 64 | 7.9008 |

## Caveat

Synthetic PTB is single-lead; sqi_iSQI is a detector-agreement proxy, not true inter-lead iSQI.