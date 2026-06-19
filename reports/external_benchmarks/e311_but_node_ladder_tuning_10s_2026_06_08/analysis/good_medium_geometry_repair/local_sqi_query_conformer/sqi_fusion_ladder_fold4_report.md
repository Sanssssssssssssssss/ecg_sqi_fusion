# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 23:46:33
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `4`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.990596 | 0.660539 | 0.990431 | 1 | 0 | 0.995215 | 0 | 0 | 1.56578 |
| L0_waveform_only | clean_test | 0.970692 | 0.945946 | 0.967988 | 0.949367 | 1 | 0.958677 | 0 | 0 | 1.38702 |
| L1_oracle_sqi_diag | clean_val | 0.996865 | 0.665145 | 1 | 1 | 0 | 1 | 0 | 0 | 1.50581 |
| L1_oracle_sqi_diag | clean_test | 0.995311 | 0.99075 | 0.993902 | 1 | 1 | 0.996951 | 0 | 0 | 1.50386 |
| L2_pred_sqi_stopgrad | clean_val | 0.978056 | 0.651442 | 0.971292 | 1 | 0 | 0.985646 | 0 | 0 | 1.62247 |
| L2_pred_sqi_stopgrad | clean_test | 0.882767 | 0.842334 | 0.849085 | 0.987342 | 1 | 0.918214 | 0 | 0 | 1.50844 |
| L3_pred_sqi_e2e | clean_val | 0.874608 | 0.580686 | 0.813397 | 1 | 0 | 0.906699 | 0 | 0 | 1.65842 |
| L3_pred_sqi_e2e | clean_test | 0.89449 | 0.854482 | 0.862805 | 1 | 1 | 0.931402 | 0 | 0 | 1.70378 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | 0.677049 | -0.621292 | 0.859732 | 0.632905 | 0.83082 | 0 | 0.98012 | 0.877004 | 0.56579 | 0.122209 |
| L1_oracle_sqi_diag | 0.625912 | -0.642232 | 0.828428 | 0.626326 | 0.781059 | 0 | 0.970595 | 0.910324 | 0.444168 | -0.400782 |
| L2_pred_sqi_stopgrad | 0.673167 | -0.540045 | 0.831697 | 0.703934 | 0.838598 | 0 | 0.977409 | 0.920314 | 0.527328 | 0.022007 |
| L3_pred_sqi_e2e | 0.656233 | -0.640541 | 0.83615 | 0.714125 | 0.81677 | 0 | 0.981525 | 0.926867 | 0.423882 | -0.41695 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.