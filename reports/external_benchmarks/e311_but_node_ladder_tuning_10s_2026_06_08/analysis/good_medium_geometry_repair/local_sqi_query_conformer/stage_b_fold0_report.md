# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 23:22:32
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `0`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | clean_val | 0.95992 | 0.610985 | 0.95672 | 1 | 0 | 0.97836 | 0 | 0 | 1.50275 |
| B0_query_ce_nolocal | clean_test | 0.97223 | 0.966177 | 0.987484 | 0.892444 | 1 | 0.939964 | 0 | 0 | 1.59415 |
| B1_query_ce_local | clean_val | 0.869739 | 0.522062 | 0.854214 | 1 | 0 | 0.927107 | 0 | 0 | 1.5177 |
| B1_query_ce_local | clean_test | 0.583914 | 0.466001 | 0.972991 | 0.824889 | 0 | 0.89894 | 0 | 0 | 1.63044 |
| B2_query_hier_nolocal | clean_val | 0.98998 | 0.651591 | 0.990888 | 1 | 0 | 0.995444 | 0 | 0 | 1.43834 |
| B2_query_hier_nolocal | clean_test | 0.953469 | 0.941235 | 0.995169 | 0.787111 | 1 | 0.89114 | 0 | 0 | 1.577 |
| B3_query_hier_local | clean_val | 0.809619 | 0.478031 | 0.785877 | 1 | 0 | 0.892938 | 0 | 0 | 1.51152 |
| B3_query_hier_local | clean_test | 0.966936 | 0.959692 | 0.980896 | 0.880444 | 1 | 0.93067 | 0.00102881 | 0.00102881 | 1.53349 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | 0.667729 | -0.673889 | 0.902839 | 0.840667 | 0.920249 | -0.000907822 | 0.98977 | 0.964997 | 0.764865 | -0.185355 |
| B1_query_ce_local | 0.673204 | -0.887882 | 0.888533 | 0.735893 | 0.799738 | -0.00443043 | 0.944669 | 0.953141 | -0.479286 | -0.0482748 |
| B2_query_hier_nolocal | 0.662502 | -0.769082 | 0.900978 | 0.639866 | 0.902773 | -0.00246407 | 0.98662 | 0.938584 | -0.45179 | -0.251812 |
| B3_query_hier_local | 0.666667 | -0.528837 | 0.880026 | 0.873159 | 0.849057 | -0.00224435 | 0.9709 | 0.728723 | -0.424852 | -0.217234 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.