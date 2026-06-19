# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 23:39:08
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `1`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.966872 | 0.642417 | 0.924805 | 0.987835 | 0 | 0.95632 | 0.00194868 | 0.00194868 | 1.68404 |
| L0_waveform_only | clean_test | 0.600392 | 0.399734 | 0.454117 | 0.991013 | 0 | 0.722565 | 0.00228534 | 0.00228534 | 1.50133 |
| L1_oracle_sqi_diag | clean_val | 0.975317 | 0.648926 | 0.982422 | 0.971776 | 0 | 0.977099 | 0.00129912 | 0.00129912 | 1.61232 |
| L1_oracle_sqi_diag | clean_test | 0.912504 | 0.599163 | 0.882208 | 0.993409 | 0 | 0.937808 | 0.000652955 | 0.000652955 | 1.38777 |
| L2_pred_sqi_stopgrad | clean_val | 0.901916 | 0.597698 | 0.99707 | 0.854501 | 0 | 0.925786 | 0.000974342 | 0.000974342 | 1.63054 |
| L2_pred_sqi_stopgrad | clean_test | 0.939602 | 0.616513 | 0.95221 | 0.905932 | 0 | 0.929071 | 0.000489716 | 0.000489716 | 1.37662 |
| L3_pred_sqi_e2e | clean_val | 0.89672 | 0.594011 | 0.999023 | 0.845742 | 0 | 0.922383 | 0.000324781 | 0.000324781 | 1.72282 |
| L3_pred_sqi_e2e | clean_test | 0.921645 | 0.604341 | 0.911151 | 0.94967 | 0 | 0.930411 | 0.000489716 | 0.000489716 | 1.45664 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | 0.370377 | 0.451112 | 0.464857 | 0.661703 | 0.848399 | 0 | 0.808248 | 0.699016 | 0.41472 | -0.192695 |
| L1_oracle_sqi_diag | 0.374162 | 0.654256 | 0.283912 | 0.780081 | 0.720622 | 0 | 0.775682 | 0.861217 | 0.709027 | -0.180655 |
| L2_pred_sqi_stopgrad | 0.420765 | 0.521941 | 0.285127 | 0.698711 | 0.736092 | 0 | 0.780207 | 0.691553 | 0.687641 | 0.115996 |
| L3_pred_sqi_e2e | 0.38572 | 0.537183 | 0.351391 | 0.734766 | 0.735751 | 0 | 0.755945 | 0.660143 | 0.69504 | -0.169247 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.