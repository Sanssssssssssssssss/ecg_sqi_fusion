# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 23:24:38
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `1`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | clean_val | 0.975641 | 0.648564 | 0.966797 | 0.980049 | 0 | 0.973423 | 0.000324781 | 0.000324781 | 1.71054 |
| B0_query_ce_nolocal | clean_test | 0.778485 | 0.510445 | 0.698901 | 0.991013 | 0 | 0.844957 | 0.00163239 | 0.00163239 | 1.51235 |
| B1_query_ce_local | clean_val | 0.91848 | 0.60948 | 0.999023 | 0.878345 | 0 | 0.938684 | 0.00227347 | 0.00227347 | 1.72145 |
| B1_query_ce_local | clean_test | 0.953151 | 0.628331 | 0.963877 | 0.924506 | 0 | 0.944191 | 0.00359125 | 0.00359125 | 1.3689 |
| B2_query_hier_nolocal | clean_val | 0.948685 | 0.629999 | 0.998047 | 0.924088 | 0 | 0.961067 | 0.00129912 | 0.00129912 | 1.70615 |
| B2_query_hier_nolocal | clean_test | 0.919687 | 0.602946 | 0.90958 | 0.946675 | 0 | 0.928128 | 0.00114267 | 0.00114267 | 1.49241 |
| B3_query_hier_local | clean_val | 0.881455 | 0.58385 | 0.999023 | 0.822871 | 0 | 0.910947 | 0.000324781 | 0.000324781 | 1.7078 |
| B3_query_hier_local | clean_test | 0.940581 | 0.615299 | 0.977788 | 0.841222 | 0 | 0.909505 | 0.00114267 | 0.00114267 | 1.4179 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | 0.470815 | 0.265337 | 0.500826 | 0.742185 | 0.87236 | 0 | 0.793415 | 0.83036 | 0.601666 | -0.104657 |
| B1_query_ce_local | 0.535269 | 0.509326 | 0.427617 | 0.714578 | 0.842685 | 0 | 0.801469 | 0.538525 | 0.505851 | 0.0630716 |
| B2_query_hier_nolocal | 0.561949 | 0.54015 | 0.4681 | 0.750721 | 0.900605 | 0 | 0.801048 | 0.839308 | 0.64502 | -0.13269 |
| B3_query_hier_local | 0.48978 | 0.706029 | 0.289682 | 0.774216 | 0.790292 | 0 | 0.792641 | 0.832517 | 0.5673 | -0.0527037 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.