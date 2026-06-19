# LocalSQI Query Conformer stage_d Report

- Created: 2026-06-19 23:50:07
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `1`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | clean_val | 0.958103 | 0.636067 | 0.943359 | 0.96545 | 0 | 0.954405 | 0.0016239 | 0.0016239 | 1.72796 |
| D0_no_pretrain | clean_test | 0.553053 | 0.368821 | 0.387032 | 0.996405 | 0 | 0.691718 | 0.000816193 | 0.000816193 | 1.51673 |
| D1_masked_recon_pretrain | clean_val | 0.953232 | 0.63281 | 0.951172 | 0.954258 | 0 | 0.952715 | 0.00194868 | 0.00194868 | 1.68611 |
| D1_masked_recon_pretrain | clean_test | 0.684786 | 0.452174 | 0.571461 | 0.987418 | 0 | 0.779439 | 0.00130591 | 0.00130591 | 1.48575 |
| D2_masked_factor_pretrain | clean_val | 0.980513 | 0.652309 | 0.987305 | 0.977129 | 0 | 0.982217 | 0.000324781 | 0.000324781 | 1.69827 |
| D2_masked_factor_pretrain | clean_test | 0.923604 | 0.606429 | 0.906215 | 0.970042 | 0 | 0.938128 | 0.000489716 | 0.000489716 | 1.56352 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | 0.212201 | 0.404702 | 0.27341 | 0.686145 | 0.733858 | 0 | 0.796357 | 0.621145 | 0.631053 | 0.164282 |
| D1_masked_recon_pretrain | 0.227315 | 0.448322 | 0.33894 | 0.720663 | 0.6896 | 0 | 0.772237 | 0.570374 | 0.460076 | 0.196182 |
| D2_masked_factor_pretrain | 0.482307 | 0.427665 | 0.476529 | 0.754759 | 0.849746 | 0 | 0.77988 | 0.825019 | 0.617155 | 0.142566 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.