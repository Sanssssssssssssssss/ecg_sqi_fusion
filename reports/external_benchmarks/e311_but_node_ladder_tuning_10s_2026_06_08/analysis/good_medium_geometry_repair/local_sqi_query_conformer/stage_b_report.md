# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 22:52:46
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B3_query_hier_local | clean_val | 0.00555556 | 0.00453515 | 0 | 0 | 1 | 0 | 0.810056 | 0.810056 | 0.994434 |
| B3_query_hier_local | clean_test | 0.388889 | 0.271132 | 0 | 0.166667 | 1 | 0.0833333 | 0.583333 | 0.583333 | 1.24373 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B3_query_hier_local | 0.58658 | 0.0459644 | 0.594402 | 0.38323 | -0.535963 | 0 | 0.891729 | -0.863337 | 0.287921 | -0.0577 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.