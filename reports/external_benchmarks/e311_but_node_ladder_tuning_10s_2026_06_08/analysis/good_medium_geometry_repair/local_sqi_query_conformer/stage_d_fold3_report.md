# LocalSQI Query Conformer stage_d Report

- Created: 2026-06-19 23:54:46
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `3`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | clean_val | 0.931472 | 0.608025 | 0.960105 | 0.853206 | 0 | 0.906655 | 0.00144439 | 0.00144439 | 1.34686 |
| D0_no_pretrain | clean_test | 0.896349 | 0.585264 | 0.862776 | 1 | 0 | 0.931388 | 0 | 0 | 1.50321 |
| D1_masked_recon_pretrain | clean_val | 0.909003 | 0.594318 | 0.896975 | 0.941881 | 0 | 0.919428 | 0.000802439 | 0.000802439 | 1.29624 |
| D1_masked_recon_pretrain | clean_test | 0.749117 | 0.488983 | 0.665615 | 1 | 0 | 0.832808 | 0 | 0 | 1.45625 |
| D2_masked_factor_pretrain | clean_val | 0.90804 | 0.594463 | 0.89018 | 0.95686 | 0 | 0.92352 | 0.00208634 | 0.00208634 | 1.51022 |
| D2_masked_factor_pretrain | clean_test | 0.687868 | 0.451552 | 0.583596 | 1 | 0 | 0.791798 | 0 | 0 | 1.65336 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | 0.261305 | 0.418942 | 0.357169 | 0.823902 | 0.883317 | 0 | 0.757864 | 0.861152 | 0.638353 | 0.0977649 |
| D1_masked_recon_pretrain | 0.000430585 | 0.139934 | 0.204006 | 0.754121 | 0.797666 | 0 | 0.688851 | 0.750784 | 0.445616 | 0.218295 |
| D2_masked_factor_pretrain | 0.112452 | 0.342264 | 0.265399 | 0.810894 | 0.800276 | 0 | 0.606165 | 0.776411 | 0.649726 | 0.107177 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.