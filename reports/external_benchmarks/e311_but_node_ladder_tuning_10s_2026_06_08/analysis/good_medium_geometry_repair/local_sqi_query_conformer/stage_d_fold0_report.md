# LocalSQI Query Conformer stage_d Report

- Created: 2026-06-19 23:48:12
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | clean_val | 0.981964 | 0.639979 | 0.981777 | 1 | 0 | 0.990888 | 0 | 0 | 1.52069 |
| D0_no_pretrain | clean_test | 0.964243 | 0.955573 | 0.994949 | 0.839111 | 1 | 0.91703 | 0 | 0 | 1.49789 |
| D1_masked_recon_pretrain | clean_val | 0.893788 | 0.542368 | 0.881549 | 1 | 0 | 0.940774 | 0 | 0 | 1.4424 |
| D1_masked_recon_pretrain | clean_test | 0.970465 | 0.964434 | 0.974747 | 0.909778 | 1 | 0.942263 | 0.000293945 | 0.000293945 | 1.60778 |
| D2_masked_factor_pretrain | clean_val | 0.93988 | 0.587713 | 0.933941 | 1 | 0 | 0.96697 | 0 | 0 | 1.67741 |
| D2_masked_factor_pretrain | clean_test | 0.973344 | 0.967843 | 0.979578 | 0.913778 | 1 | 0.946678 | 0.000440917 | 0.000440917 | 1.61636 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | 0.680295 | -0.817244 | 0.908811 | 0.864309 | 0.896504 | -0.0010475 | 0.98744 | 0.905231 | -0.38099 | -0.221192 |
| D1_masked_recon_pretrain | 0.645796 | -0.270151 | 0.864645 | 0.858834 | 0.906979 | -0.00705059 | 0.984471 | 0.463866 | -0.132029 | 0.243116 |
| D2_masked_factor_pretrain | 0.687512 | -0.599279 | 0.911358 | 0.843097 | 0.907918 | -0.000633192 | 0.992331 | 0.954974 | 0.505354 | 0.214219 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.