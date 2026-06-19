# LocalSQI Query Conformer stage_c Report

- Created: 2026-06-19 23:32:26
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | clean_val | 0.977956 | 0.634403 | 0.977221 | 1 | 0 | 0.98861 | 0 | 0 | 1.45526 |
| C0_query_hier_local | clean_test | 0.941581 | 0.924584 | 0.997365 | 0.725778 | 1 | 0.861571 | 0 | 0 | 1.47467 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | 0.643183 | -0.728609 | 0.90456 | 0.737015 | 0.916184 | 0.0020031 | 0.991036 | 0.962702 | -0.188875 | -0.213814 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.