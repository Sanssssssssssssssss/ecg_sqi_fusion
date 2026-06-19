# LocalSQI Query Conformer stage_c Report

- Created: 2026-06-19 23:35:03
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `4`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | clean_val | 0.852665 | 0.566329 | 0.779904 | 1 | 0 | 0.889952 | 0 | 0 | 1.55752 |
| C0_query_hier_local | clean_test | 0.875733 | 0.836864 | 0.838415 | 1 | 1 | 0.919207 | 0 | 0 | 1.48369 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_query_hier_local | 0.641389 | -0.565312 | 0.82006 | 0.660698 | 0.842725 | 0 | 0.981284 | 0.890196 | 0.533782 | -0.56426 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.