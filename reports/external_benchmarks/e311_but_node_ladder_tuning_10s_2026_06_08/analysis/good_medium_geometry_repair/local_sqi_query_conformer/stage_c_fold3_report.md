# LocalSQI Query Conformer stage_c Report

- Created: 2026-06-19 23:34:22
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `3`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | clean_val | 0.854919 | 0.558787 | 0.806006 | 0.988616 | 0 | 0.897311 | 0.0012839 | 0.0012839 | 1.3649 |
| C0_query_hier_local | clean_test | 0.727915 | 0.475963 | 0.637224 | 1 | 0 | 0.818612 | 0 | 0 | 1.47679 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | 0.157927 | 0.260901 | 0.206108 | 0.779705 | 0.866119 | 0 | 0.726821 | 0.794685 | 0.419182 | -0.235588 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.