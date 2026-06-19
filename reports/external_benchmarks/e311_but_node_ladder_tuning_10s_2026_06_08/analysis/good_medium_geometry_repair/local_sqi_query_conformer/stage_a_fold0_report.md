# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 23:10:56
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.973948 | 0.628467 | 0.974943 | 0.983051 | 0 | 0.978997 | 0 | 0 | 1.42343 |
| A0_nohi_noquery | clean_test | 0.930157 | 0.908211 | 0.996267 | 0.673333 | 1 | 0.8348 | 0 | 0 | 1.33587 |
| A1_hi_noquery | clean_val | 0.955912 | 0.606118 | 0.952164 | 1 | 0 | 0.976082 | 0 | 0 | 1.42766 |
| A1_hi_noquery | clean_test | 0.952912 | 0.941289 | 0.984409 | 0.806222 | 1 | 0.895316 | 0 | 0 | 1.44498 |
| A2_nohi_query | clean_val | 0.963928 | 0.615968 | 0.961276 | 1 | 0 | 0.980638 | 0 | 0 | 1.40633 |
| A2_nohi_query | clean_test | 0.968608 | 0.961539 | 0.988362 | 0.873333 | 1 | 0.930848 | 0 | 0 | 1.50422 |
| A3_hi_query | clean_val | 0.87976 | 0.530294 | 0.865604 | 1 | 0 | 0.932802 | 0 | 0 | 1.48553 |
| A3_hi_query | clean_test | 0.573976 | 0.454262 | 0.980018 | 0.763111 | 0 | 0.871564 | 0 | 0 | 1.57209 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.628231 | -0.587865 | 0.899972 | 0.757762 | 0.760586 | -0.00377285 | 0.990109 | 0.951147 | -0.406956 | -0.227348 |
| A1_hi_noquery | 0.642779 | -0.779817 | 0.901319 | 0.841341 | 0.877581 | -0.000885126 | 0.989755 | 0.971762 | 0.103831 | -0.220459 |
| A2_nohi_query | 0.676561 | -0.732551 | 0.872682 | 0.852086 | 0.866313 | -0.00111529 | 0.984047 | 0.923906 | -0.340222 | 0.164113 |
| A3_hi_query | 0.671728 | -0.801797 | 0.868046 | 0.678219 | 0.850278 | -0.0023136 | 0.92974 | 0.915842 | -0.511715 | -0.23407 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.