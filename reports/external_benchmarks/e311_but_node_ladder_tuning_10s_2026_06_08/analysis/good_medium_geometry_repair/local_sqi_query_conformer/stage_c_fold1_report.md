# LocalSQI Query Conformer stage_c Report

- Created: 2026-06-19 23:33:01
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `1`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | clean_val | 0.893147 | 0.591036 | 0.975586 | 0.852068 | 0 | 0.913827 | 0.000324781 | 0.000324781 | 1.65233 |
| C0_query_hier_local | clean_test | 0.806889 | 0.526458 | 0.751627 | 0.954464 | 0 | 0.853045 | 0.000326477 | 0.000326477 | 1.37659 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | 0.266429 | 0.444082 | 0.240258 | 0.682305 | 0.692671 | 0 | 0.681674 | 0.662964 | 0.48431 | 0.0185374 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.