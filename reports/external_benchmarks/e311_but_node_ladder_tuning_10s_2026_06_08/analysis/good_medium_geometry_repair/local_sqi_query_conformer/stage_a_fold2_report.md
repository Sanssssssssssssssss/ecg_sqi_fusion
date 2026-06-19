# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 23:15:39
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `2`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.844059 | 0.824824 | 0.728448 | 1 | 1 | 0.864224 | 0 | 0 | 1.63287 |
| A0_nohi_noquery | clean_test | 0.988255 | 0.657675 | 0.97411 | 0.994642 | 0 | 0.984376 | 0.00033557 | 0.00033557 | 1.68225 |
| A1_hi_noquery | clean_val | 0.997525 | 0.996222 | 0.99569 | 1 | 1 | 0.997845 | 0 | 0 | 1.60624 |
| A1_hi_noquery | clean_test | 0.920134 | 0.608234 | 1 | 0.884072 | 0 | 0.942036 | 0 | 0 | 1.73541 |
| A2_nohi_query | clean_val | 0.990099 | 0.985197 | 0.982759 | 1 | 1 | 0.991379 | 0 | 0 | 1.56862 |
| A2_nohi_query | clean_test | 0.941275 | 0.623078 | 1 | 0.914759 | 0 | 0.957379 | 0 | 0 | 1.78773 |
| A3_hi_query | clean_val | 0.933168 | 0.91086 | 0.896552 | 1 | 0.974576 | 0.948276 | 0 | 0 | 1.63036 |
| A3_hi_query | clean_test | 0.986577 | 0.656295 | 0.990291 | 0.9849 | 0 | 0.987596 | 0 | 0 | 1.6525 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | -0.00186552 | 0.444007 | -0.157443 | 0.575263 | 0.913151 | 0 | 0.895125 | 0.822329 | 0.178488 | -0.41659 |
| A1_hi_noquery | 0.10345 | 0.762851 | 0.0386368 | 0.641746 | 0.883488 | 0 | 0.836681 | 0.869553 | 0.687983 | -0.202843 |
| A2_nohi_query | 0.0146689 | 0.290593 | -0.128085 | 0.57223 | 0.900103 | 0 | 0.877609 | 0.862795 | 0.515826 | 0.305087 |
| A3_hi_query | 0.130004 | 0.421737 | 0.135324 | 0.644239 | 0.892099 | 0 | 0.894868 | 0.862033 | -0.000664171 | -0.193235 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.