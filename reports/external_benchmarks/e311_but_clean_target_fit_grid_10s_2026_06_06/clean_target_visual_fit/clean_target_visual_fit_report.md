# CleanBUT-Core Visual Fit

This report projects PTB synthetic audit samples into a PCA fitted only on the CleanBUT train-target core.
It is CPU-only and uses signal-derived morphology/QRS features, not model predictions.

- Common feature count: `31`
- PCA explained variance: `0.619`, `0.143`

## Figures

- `figures/ptb_vs_cleanbut_pca_top_rules.png`: CleanBUT-Core background with top PTB synthetic variants overlaid.
- `figures/ptb_vs_cleanbut_best_rule.png`: single best current rule in the same PCA space.
- `figures/ptb_vs_cleanbut_centroid_shift.png`: class centroid shifts from CleanBUT to synthetic.
- `figures/clean_target_distance_bars.png`: class-wise distance bars from the no-training scan.

## Current Read

The closest no-training candidates are mostly `qrs_confound` and `visible_unusable` bad-subtype rules with `bw_badstrong` overlay and a protected medium guard. The medium cluster is relatively easy to match; good and bad remain the larger domain-gap contributors in this 31-feature common projection.

## Top Variants

| rank | variant_id | clean_target_score | clean_good_distance | clean_medium_distance | clean_bad_distance | clean_domain_separability_bal_acc | mean_centroid_distance_common_pca |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mg_qrs_confound_m_guard_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5170 | 0.6241 | 0.2971 | 0.6347 | 0.9204 | 6.2650 |
| 2 | mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5174 | 0.6241 | 0.3002 | 0.6311 | 0.9283 | 6.0726 |
| 3 | mg_visible_unusable_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75 | 0.5177 | 0.6241 | 0.3022 | 0.6308 | 0.9204 | 6.0653 |
| 4 | mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5185 | 0.6241 | 0.3022 | 0.6320 | 0.9281 | 6.0413 |
| 5 | mg_visible_unusable_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5186 | 0.6241 | 0.3002 | 0.6317 | 0.9440 | 5.9991 |
| 6 | mg_contact_lowamp_m_guard_badguard__bw_badstrong__cw1p00_1p62_1p78 | 0.5187 | 0.6241 | 0.2971 | 0.6449 | 0.8889 | 6.1111 |
| 7 | mg_visible_unusable_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5191 | 0.6241 | 0.3002 | 0.6331 | 0.9438 | 6.0197 |
| 8 | mg_balanced_or_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5191 | 0.6241 | 0.2971 | 0.6393 | 0.9286 | 6.1109 |
| 9 | mg_qrs_confound_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70 | 0.5193 | 0.6241 | 0.2971 | 0.6345 | 0.9599 | 6.2171 |

## Caveat

The original CleanBUT atlas uses 64 morphology/SQI/frequency/QRS features. Current synthetic audit files contain 31 common morphology/QRS features, so this visualization is the fair common-feature view. A stricter 64-feature comparison requires computing SQI/frequency features for synthetic samples as the next CPU-only audit.