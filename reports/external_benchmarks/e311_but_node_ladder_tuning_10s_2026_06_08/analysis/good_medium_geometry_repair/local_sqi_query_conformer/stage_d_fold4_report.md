# LocalSQI Query Conformer stage_d Report

- Created: 2026-06-19 23:57:26
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `4`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | clean_val | 0.952978 | 0.633656 | 0.933014 | 1 | 0 | 0.966507 | 0 | 0 | 1.61809 |
| D0_no_pretrain | clean_test | 0.875733 | 0.836864 | 0.838415 | 1 | 1 | 0.919207 | 0 | 0 | 1.4698 |
| D1_masked_recon_pretrain | clean_val | 0.799373 | 0.531862 | 0.698565 | 1 | 0 | 0.849282 | 0 | 0 | 1.80933 |
| D1_masked_recon_pretrain | clean_test | 0.87456 | 0.83581 | 0.83689 | 1 | 1 | 0.918445 | 0 | 0 | 1.58983 |
| D2_masked_factor_pretrain | clean_val | 0.962382 | 0.640266 | 0.947368 | 1 | 0 | 0.973684 | 0 | 0 | 1.87209 |
| D2_masked_factor_pretrain | clean_test | 0.955451 | 0.92131 | 0.95122 | 0.924051 | 1 | 0.937635 | 0 | 0 | 1.64738 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | 0.654414 | -0.653951 | 0.82338 | 0.710285 | 0.825826 | 0 | 0.973988 | 0.927211 | 0.558822 | 0.047209 |
| D1_masked_recon_pretrain | 0.559623 | -0.401474 | 0.807958 | 0.73042 | 0.867282 | 0 | 0.981416 | 0.927703 | 0.74711 | 0.361244 |
| D2_masked_factor_pretrain | 0.609325 | -0.439133 | 0.835821 | 0.796952 | 0.842978 | 0 | 0.981443 | 0.923359 | 0.705696 | 0.22358 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.