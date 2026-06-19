# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 23:18:00
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `3`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.931311 | 0.608928 | 0.952652 | 0.872978 | 0 | 0.912815 | 0.00272829 | 0.00272829 | 1.26683 |
| A0_nohi_noquery | clean_test | 0.776207 | 0.838544 | 0.701893 | 0.995327 | 1 | 0.84861 | 0 | 0 | 1.4783 |
| A1_hi_noquery | clean_val | 0.936447 | 0.615647 | 0.928321 | 0.958658 | 0 | 0.943489 | 0.00256781 | 0.00256781 | 1.30699 |
| A1_hi_noquery | clean_test | 0.725559 | 0.807461 | 0.632492 | 1 | 1 | 0.816246 | 0 | 0 | 1.45345 |
| A2_nohi_query | clean_val | 0.929385 | 0.609967 | 0.916922 | 0.963451 | 0 | 0.940187 | 0.000320976 | 0.000320976 | 1.33415 |
| A2_nohi_query | clean_test | 0.75265 | 0.491163 | 0.670347 | 1 | 0 | 0.835174 | 0 | 0 | 1.47416 |
| A3_hi_query | clean_val | 0.932595 | 0.612049 | 0.927444 | 0.946675 | 0 | 0.937059 | 0.00112342 | 0.00112342 | 1.35411 |
| A3_hi_query | clean_test | 0.714959 | 0.468047 | 0.619874 | 1 | 0 | 0.809937 | 0 | 0 | 1.53379 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.0572361 | 0.325952 | 0.300269 | 0.815772 | 0.779798 | 0 | 0.655487 | 0.778 | 0.499125 | -0.291433 |
| A1_hi_noquery | 0.0646993 | 0.586855 | 0.320719 | 0.78196 | 0.863359 | 0 | 0.702557 | 0.907078 | 0.651465 | 0.160512 |
| A2_nohi_query | 0.121799 | 0.229859 | 0.28989 | 0.81236 | 0.856273 | 0 | 0.737784 | 0.821984 | 0.587499 | 0.0375006 |
| A3_hi_query | 0.0970133 | 0.398154 | 0.318191 | 0.80075 | 0.831362 | 0 | 0.698349 | 0.831614 | 0.569346 | -0.0773551 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.