# LocalSQI Query Conformer sqi_fusion_ladder Report

- Created: 2026-06-19 23:43:53
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `3`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | clean_val | 0.941422 | 0.618134 | 0.944761 | 0.932295 | 0 | 0.938528 | 0 | 0 | 1.35518 |
| L0_waveform_only | clean_test | 0.837456 | 0.545017 | 0.783912 | 1 | 0 | 0.891956 | 0 | 0 | 1.49975 |
| L1_oracle_sqi_diag | clean_val | 0.956187 | 0.629288 | 0.972819 | 0.910725 | 0 | 0.941772 | 0.000160488 | 0.000160488 | 1.29284 |
| L1_oracle_sqi_diag | clean_test | 0.987044 | 0.655631 | 0.985804 | 0.995327 | 0 | 0.990566 | 0 | 0 | 1.44634 |
| L2_pred_sqi_stopgrad | clean_val | 0.912053 | 0.589288 | 0.959009 | 0.783703 | 0 | 0.871356 | 0 | 0 | 1.27281 |
| L2_pred_sqi_stopgrad | clean_test | 0.849234 | 0.5524 | 0.802839 | 0.990654 | 0 | 0.896747 | 0 | 0 | 1.48657 |
| L3_pred_sqi_e2e | clean_val | 0.908843 | 0.589106 | 0.938404 | 0.828041 | 0 | 0.883222 | 0 | 0 | 1.29501 |
| L3_pred_sqi_e2e | clean_test | 0.859835 | 0.55932 | 0.818612 | 0.985981 | 0 | 0.902297 | 0 | 0 | 1.46248 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0_waveform_only | 0.225309 | 0.46896 | 0.29161 | 0.802696 | 0.878989 | 0 | 0.706469 | 0.847175 | 0.574548 | 0.0708452 |
| L1_oracle_sqi_diag | 0.298914 | 0.509087 | 0.367143 | 0.836346 | 0.864139 | 0 | 0.707772 | 0.861362 | 0.69363 | -0.161756 |
| L2_pred_sqi_stopgrad | 0.259494 | 0.262083 | 0.321382 | 0.828577 | 0.825885 | 0 | 0.732685 | 0.805238 | 0.537851 | 0.0931225 |
| L3_pred_sqi_e2e | 0.158768 | 0.0933731 | 0.137269 | 0.765528 | 0.832046 | 0 | 0.698138 | 0.627194 | 0.539101 | -0.212074 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.