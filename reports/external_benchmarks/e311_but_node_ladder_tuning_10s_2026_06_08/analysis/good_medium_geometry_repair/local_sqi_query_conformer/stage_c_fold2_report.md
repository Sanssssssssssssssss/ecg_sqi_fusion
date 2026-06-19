# LocalSQI Query Conformer stage_c Report

- Created: 2026-06-19 23:33:43
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `2`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | clean_val | 0.970297 | 0.957291 | 0.952586 | 0.981481 | 1 | 0.967034 | 0 | 0 | 1.49746 |
| C0_query_hier_local | clean_test | 0.931544 | 0.616467 | 0.998921 | 0.90112 | 0 | 0.950021 | 0.000671141 | 0.000671141 | 1.72585 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | -0.152382 | -0.0562861 | -0.125256 | 0.551295 | 0.913845 | 0 | 0.896945 | 0.758305 | 0.0440989 | -0.343467 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.