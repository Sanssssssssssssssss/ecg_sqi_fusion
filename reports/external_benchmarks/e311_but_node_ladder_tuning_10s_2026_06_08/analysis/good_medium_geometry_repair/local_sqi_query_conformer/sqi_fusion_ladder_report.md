# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 22:52:32
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.846667 | 0.535119 | 0.8375 | 0.898305 | 0 | 0.867903 | 0.0100334 | 0.0100334 | 1.11138 |
| L0_waveform_only | clean_test | 0.84 | 0.832274 | 0.96 | 0.56 | 1 | 0.76 | 0.01 | 0.01 | 1.32961 |
| L1_oracle_sqi_diag | clean_val | 0.95 | 0.619743 | 0.954167 | 0.949153 | 0 | 0.95166 | 0.00334448 | 0.00334448 | 1.09732 |
| L1_oracle_sqi_diag | clean_test | 0.82 | 0.807239 | 0.95 | 0.51 | 1 | 0.73 | 0.035 | 0.035 | 1.30913 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | 0.341899 | -0.792836 | 0.757373 | 0.382956 | -0.520066 | 0 | 0.892191 | -0.205684 | -0.421819 | 0.160763 |
| L1_oracle_sqi_diag | -0.0620981 | 0.599936 | 0.83195 | 0.345448 | 0.0393839 | 0 | 0.930275 | 0.143098 | 0.316594 | 0.228457 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.