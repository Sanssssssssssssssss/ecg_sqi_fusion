# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 22:52:17
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.583333 | 0.365218 | 0.583333 | 0.59322 | 0 | 0.588277 | 0.0802676 | 0.0802676 | 0.986165 |
| A0_nohi_noquery | clean_test | 0.806667 | 0.791288 | 0.91 | 0.51 | 1 | 0.71 | 0.1 | 0.1 | 1.28221 |
| A1_hi_noquery | clean_val | 0.906667 | 0.579511 | 0.904167 | 0.932203 | 0 | 0.918185 | 0 | 0 | 0.916726 |
| A1_hi_noquery | clean_test | 0.8 | 0.784231 | 0.96 | 0.44 | 1 | 0.7 | 0.01 | 0.01 | 1.18586 |
| A2_nohi_query | clean_val | 0.84 | 0.447491 | 0.975 | 0.305085 | 0 | 0.640042 | 0.00334448 | 0.00334448 | 1.06391 |
| A2_nohi_query | clean_test | 0.7 | 0.611514 | 1 | 0.1 | 1 | 0.55 | 0.09 | 0.09 | 1.30665 |
| A3_hi_query | clean_val | 0.876667 | 0.538534 | 0.933333 | 0.661017 | 0 | 0.797175 | 0.0133779 | 0.0133779 | 1.05713 |
| A3_hi_query | clean_test | 0.743333 | 0.703128 | 0.96 | 0.27 | 1 | 0.615 | 0.04 | 0.04 | 1.37105 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.290614 | -0.746494 | 0.760554 | 0.204685 | -0.644973 | 0 | 0.786327 | 0.242128 | -0.498289 | -0.210655 |
| A1_hi_noquery | 0.597751 | -0.642126 | -0.567243 | 0.12596 | -0.0268551 | 0 | 0.929827 | 0.937918 | -0.483042 | 0.133775 |
| A2_nohi_query | 0.581686 | -0.500659 | -0.393643 | -0.455294 | 0.489064 | 0 | 0.908208 | -0.695102 | -0.498947 | 0.227797 |
| A3_hi_query | 0.619016 | -0.82611 | 0.811432 | 0.303952 | -0.651942 | 0 | 0.918687 | -0.816661 | -0.501529 | -0.198705 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.