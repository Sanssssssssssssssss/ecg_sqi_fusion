# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 23:36:58
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.913828 | 0.560893 | 0.904328 | 1 | 0 | 0.952164 | 0 | 0 | 1.51464 |
| L0_waveform_only | clean_test | 0.568775 | 0.449634 | 0.975626 | 0.747111 | 0 | 0.861368 | 0 | 0 | 1.61395 |
| L1_oracle_sqi_diag | clean_val | 0.983968 | 0.642824 | 0.984055 | 1 | 0 | 0.992027 | 0 | 0 | 1.49971 |
| L1_oracle_sqi_diag | clean_test | 0.97093 | 0.964251 | 0.994291 | 0.872444 | 1 | 0.933368 | 0 | 0 | 1.55589 |
| L2_pred_sqi_stopgrad | clean_val | 0.947896 | 0.596712 | 0.943052 | 1 | 0 | 0.971526 | 0 | 0 | 1.48564 |
| L2_pred_sqi_stopgrad | clean_test | 0.96062 | 0.951001 | 0.991875 | 0.828 | 1 | 0.909938 | 0 | 0 | 1.6024 |
| L3_pred_sqi_e2e | clean_val | 0.839679 | 0.499029 | 0.820046 | 1 | 0 | 0.910023 | 0 | 0 | 1.58017 |
| L3_pred_sqi_e2e | clean_test | 0.574162 | 0.457607 | 0.963549 | 0.797333 | 0 | 0.880441 | 0 | 0 | 1.7191 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | 0.646835 | -0.612333 | 0.835717 | 0.279662 | 0.835399 | 9.58012e-05 | 0.886313 | 0.959303 | -0.407789 | -0.150739 |
| L1_oracle_sqi_diag | 0.647358 | -0.791722 | 0.892606 | 0.841221 | 0.896256 | -0.00300573 | 0.986743 | 0.96727 | -0.0978595 | -0.205124 |
| L2_pred_sqi_stopgrad | 0.664341 | -0.818796 | 0.889727 | 0.817652 | 0.909523 | -0.00224433 | 0.987157 | 0.957515 | -0.139547 | 0.17026 |
| L3_pred_sqi_e2e | 0.666239 | -0.791338 | 0.703773 | 0.41475 | 0.85457 | -0.00216935 | 0.854359 | 0.852441 | -0.437665 | -0.199128 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.