# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 23:27:05
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `2`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | clean_val | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.57492 |
| B0_query_ce_nolocal | clean_test | 0.927181 | 0.61314 | 1 | 0.894301 | 0 | 0.947151 | 0 | 0 | 1.75861 |
| B1_query_ce_local | clean_val | 0.970297 | 0.957817 | 0.948276 | 1 | 1 | 0.974138 | 0 | 0 | 1.49512 |
| B1_query_ce_local | clean_test | 0.957047 | 0.634298 | 0.992449 | 0.941062 | 0 | 0.966755 | 0 | 0 | 1.73364 |
| B2_query_hier_nolocal | clean_val | 0.990099 | 0.985197 | 0.982759 | 1 | 1 | 0.991379 | 0 | 0 | 1.60703 |
| B2_query_hier_nolocal | clean_test | 0.948322 | 0.628266 | 1 | 0.924988 | 0 | 0.962494 | 0.00033557 | 0.00033557 | 1.76666 |
| B3_query_hier_local | clean_val | 0.987624 | 0.98162 | 0.978448 | 1 | 1 | 0.989224 | 0 | 0 | 1.68123 |
| B3_query_hier_local | clean_test | 0.947315 | 0.627372 | 0.998921 | 0.924014 | 0 | 0.961467 | 0 | 0 | 1.79251 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | 0.0060967 | 0.541589 | -0.237182 | 0.580637 | 0.898774 | 0 | 0.823187 | 0.871662 | 0.474911 | -0.105446 |
| B1_query_ce_local | -0.127986 | -0.161248 | 0.0490612 | 0.448688 | 0.899526 | 0 | 0.888462 | 0.67799 | -0.372547 | 0.0550241 |
| B2_query_hier_nolocal | 0.0538407 | 0.443766 | -0.0452669 | 0.599271 | 0.905065 | 0 | 0.901857 | 0.854993 | 0.446747 | -0.224786 |
| B3_query_hier_local | 0.120456 | 0.515447 | -0.0515293 | 0.622878 | 0.893645 | 0 | 0.87823 | 0.856571 | 0.290518 | -0.352963 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.