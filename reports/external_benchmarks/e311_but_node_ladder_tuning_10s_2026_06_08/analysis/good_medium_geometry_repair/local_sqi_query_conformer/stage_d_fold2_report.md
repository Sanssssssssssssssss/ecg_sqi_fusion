# LocalSQI Query Conformer stage_d Report

- Created: 2026-06-19 23:52:33
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `2`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | clean_val | 0.99505 | 0.992496 | 0.991379 | 1 | 1 | 0.99569 | 0 | 0 | 1.68202 |
| D0_no_pretrain | clean_test | 0.971141 | 0.644912 | 0.998921 | 0.958597 | 0 | 0.978759 | 0.00033557 | 0.00033557 | 1.97143 |
| D1_masked_recon_pretrain | clean_val | 0.982673 | 0.974604 | 0.969828 | 1 | 1 | 0.984914 | 0 | 0 | 1.572 |
| D1_masked_recon_pretrain | clean_test | 0.946309 | 0.62665 | 0.998921 | 0.922552 | 0 | 0.960737 | 0 | 0 | 1.74784 |
| D2_masked_factor_pretrain | clean_val | 0.982673 | 0.974604 | 0.969828 | 1 | 1 | 0.984914 | 0 | 0 | 1.69878 |
| D2_masked_factor_pretrain | clean_test | 0.963087 | 0.639275 | 0.997843 | 0.947394 | 0 | 0.972618 | 0.00100671 | 0.00100671 | 1.79288 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_no_pretrain | -0.00714153 | 0.168738 | -0.10223 | 0.530171 | 0.924068 | 0 | 0.907271 | 0.860182 | 0.364649 | -0.369372 |
| D1_masked_recon_pretrain | -0.194178 | 0.110753 | -0.05838 | 0.610633 | 0.902121 | 0 | 0.874857 | 0.781509 | 0.174388 | -0.234264 |
| D2_masked_factor_pretrain | 0.191164 | 0.684317 | 0.0315382 | 0.676902 | 0.894701 | 0 | 0.861961 | 0.797341 | 0.655616 | -0.166345 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.