# LocalSQI Query Conformer stage_a Report

- Created: 2026-06-19 23:13:04
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `1`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | clean_val | 0.934719 | 0.621368 | 0.992188 | 0.906083 | 0 | 0.949135 | 0.00422215 | 0.00422215 | 1.62941 |
| A0_nohi_noquery | clean_test | 0.941561 | 0.619927 | 0.939646 | 0.946675 | 0 | 0.94316 | 0.00375449 | 0.00375449 | 1.4059 |
| A1_hi_noquery | clean_val | 0.933745 | 0.619078 | 0.986328 | 0.907543 | 0 | 0.946935 | 0.000649562 | 0.000649562 | 1.56484 |
| A1_hi_noquery | clean_test | 0.850147 | 0.55599 | 0.800314 | 0.983223 | 0 | 0.891769 | 0.000816193 | 0.000816193 | 1.32551 |
| A2_nohi_query | clean_val | 0.878857 | 0.582224 | 0.998047 | 0.819465 | 0 | 0.908756 | 0.000649562 | 0.000649562 | 1.66917 |
| A2_nohi_query | clean_test | 0.943519 | 0.618992 | 0.966569 | 0.881965 | 0 | 0.924267 | 0.00114267 | 0.00114267 | 1.34516 |
| A3_hi_query | clean_val | 0.91783 | 0.607196 | 0.952148 | 0.90073 | 0 | 0.926439 | 0.000324781 | 0.000324781 | 1.8017 |
| A3_hi_query | clean_test | 0.726902 | 0.477608 | 0.632039 | 0.980228 | 0 | 0.806134 | 0.000163239 | 0.000163239 | 1.53039 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_nohi_noquery | 0.515049 | 0.562901 | 0.37247 | 0.761263 | 0.848599 | 0 | 0.782642 | 0.830494 | 0.638958 | -0.16682 |
| A1_hi_noquery | 0.422799 | 0.585499 | 0.383171 | 0.785172 | 0.782825 | 0 | 0.777044 | 0.840821 | 0.537594 | -0.0703962 |
| A2_nohi_query | 0.441295 | 0.623325 | 0.293446 | 0.702931 | 0.747837 | 0 | 0.787403 | 0.731673 | 0.573584 | -0.0206863 |
| A3_hi_query | 0.215327 | 0.69505 | 0.234741 | 0.705085 | 0.617196 | 0 | 0.626509 | 0.567716 | 0.654541 | -0.0986545 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.