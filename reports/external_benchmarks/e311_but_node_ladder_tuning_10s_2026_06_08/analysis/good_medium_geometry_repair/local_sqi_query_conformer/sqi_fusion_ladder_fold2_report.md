# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 23:41:35
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `2`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.99505 | 0.992496 | 0.991379 | 1 | 1 | 0.99569 | 0 | 0 | 1.57692 |
| L0_waveform_only | clean_test | 0.934899 | 0.618561 | 1 | 0.905504 | 0 | 0.952752 | 0 | 0 | 1.76813 |
| L1_oracle_sqi_diag | clean_val | 0.99505 | 0.99228 | 1 | 0.962963 | 1 | 0.981481 | 0 | 0 | 1.61434 |
| L1_oracle_sqi_diag | clean_test | 0.82953 | 0.548024 | 1 | 0.752557 | 0 | 0.876279 | 0.00033557 | 0.00033557 | 1.81418 |
| L2_pred_sqi_stopgrad | clean_val | 0.987624 | 0.98162 | 0.978448 | 1 | 1 | 0.989224 | 0 | 0 | 1.67475 |
| L2_pred_sqi_stopgrad | clean_test | 0.960067 | 0.636772 | 1 | 0.942036 | 0 | 0.971018 | 0.00033557 | 0.00033557 | 1.75848 |
| L3_pred_sqi_e2e | clean_val | 0.992574 | 0.988822 | 0.987069 | 1 | 1 | 0.993534 | 0 | 0 | 1.67615 |
| L3_pred_sqi_e2e | clean_test | 0.92651 | 0.613392 | 1 | 0.893327 | 0 | 0.946663 | 0.00167785 | 0.00167785 | 1.80826 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | -0.0430842 | 0.255552 | -0.175551 | 0.543659 | 0.911521 | 0 | 0.900753 | 0.833964 | 0.441176 | 0.382416 |
| L1_oracle_sqi_diag | -0.196367 | 0.613683 | -0.264615 | 0.548497 | 0.80728 | 0 | 0.776798 | 0.754698 | 0.676619 | -0.421163 |
| L2_pred_sqi_stopgrad | 0.0561365 | 0.457237 | -0.0894551 | 0.642765 | 0.877412 | 0 | 0.866524 | 0.839992 | 0.564684 | 0.259924 |
| L3_pred_sqi_e2e | -0.08383 | 0.412905 | -0.0775654 | 0.627695 | 0.876807 | 0 | 0.852619 | 0.772951 | 0.497472 | -0.413443 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.