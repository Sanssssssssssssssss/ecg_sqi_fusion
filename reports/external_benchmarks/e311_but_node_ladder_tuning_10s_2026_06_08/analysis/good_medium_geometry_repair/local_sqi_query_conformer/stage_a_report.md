# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 22:55:35
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.947896 | 0.596712 | 0.943052 | 1 | 0 | 0.971526 | 0 | 0 | 1.51274 |
| A0_nohi_noquery | clean_test | 0.968143 | 0.9612 | 0.982653 | 0.882667 | 1 | 0.93266 | 0 | 0 | 1.51773 |
| A1_hi_noquery | clean_val | 0.943888 | 0.592164 | 0.938497 | 1 | 0 | 0.969248 | 0 | 0 | 1.5107 |
| A1_hi_noquery | clean_test | 0.968515 | 0.961586 | 0.984629 | 0.880444 | 1 | 0.932537 | 0 | 0 | 1.47628 |
| A2_nohi_query | clean_val | 0.961924 | 0.613461 | 0.958998 | 1 | 0 | 0.979499 | 0 | 0 | 1.49974 |
| A2_nohi_query | clean_test | 0.974087 | 0.968463 | 0.989021 | 0.898222 | 1 | 0.943621 | 0 | 0 | 1.64502 |
| A3_hi_query | clean_val | 0.891784 | 0.540601 | 0.879271 | 1 | 0 | 0.939636 | 0 | 0 | 1.3797 |
| A3_hi_query | clean_test | 0.5528 | 0.433246 | 0.972552 | 0.676889 | 0 | 0.82472 | 0 | 0 | 1.61552 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.694341 | -0.740815 | 0.902028 | 0.724956 | 0.921885 | -0.0031643 | 0.98984 | 0.889064 | -0.259453 | 0.222421 |
| A1_hi_noquery | 0.656938 | -0.636839 | 0.895323 | 0.888671 | 0.934846 | -0.00252436 | 0.991418 | 0.965822 | -0.0922231 | 0.208642 |
| A2_nohi_query | 0.687658 | -0.784691 | 0.901988 | 0.722276 | 0.921089 | -0.0037191 | 0.986059 | 0.636581 | -0.358815 | 0.164695 |
| A3_hi_query | 0.640385 | -0.812064 | 0.552943 | 0.612564 | 0.845903 | -9.92153e-05 | 0.917405 | 0.918519 | -0.447029 | -0.0347563 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.