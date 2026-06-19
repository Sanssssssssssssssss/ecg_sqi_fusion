# LocalSQI Query Conformer stage_b Report

- Created: 2026-06-19 23:29:20
- Policy: `margin_ge_5s_drop_outlier`
- Fold: `3`

## Metrics

| candidate | bucket | acc | macro_f1 | good_recall | medium_recall | bad_recall | gm_balanced_acc | bad_fpr_nonbad | artifact_positive_nonbad_bad_fpr | factor_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | clean_val | 0.943027 | 0.619568 | 0.945419 | 0.936489 | 0 | 0.940954 | 0.000320976 | 0.000320976 | 1.3968 |
| B0_query_ce_nolocal | clean_test | 0.826855 | 0.538074 | 0.769716 | 1 | 0 | 0.884858 | 0 | 0 | 1.51293 |
| B1_query_ce_local | clean_val | 0.920238 | 0.59729 | 0.960324 | 0.810665 | 0 | 0.885495 | 0.00112342 | 0.00112342 | 1.24295 |
| B1_query_ce_local | clean_test | 0.925795 | 0.939057 | 0.908517 | 0.976636 | 1 | 0.942576 | 0 | 0 | 1.36964 |
| B2_query_hier_nolocal | clean_val | 0.9342 | 0.611953 | 0.946734 | 0.89994 | 0 | 0.923337 | 0.00192585 | 0.00192585 | 1.30188 |
| B2_query_hier_nolocal | clean_test | 0.817432 | 0.531964 | 0.757098 | 1 | 0 | 0.878549 | 0 | 0 | 1.51413 |
| B3_query_hier_local | clean_val | 0.934842 | 0.611092 | 0.957913 | 0.87178 | 0 | 0.914846 | 0.000160488 | 0.000160488 | 1.24667 |
| B3_query_hier_local | clean_test | 0.799764 | 0.520648 | 0.733438 | 1 | 0 | 0.866719 | 0 | 0 | 1.43973 |

## Factor Recovery Corr

| candidate | qrs_visibility | qrs_band_ratio | template_corr | baseline_step | flatline_ratio | contact_loss_win_ratio | non_qrs_diff_p95 | amplitude_entropy | sqi_basSQI | detector_agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_query_ce_nolocal | 0.222541 | 0.377282 | 0.381282 | 0.805134 | 0.885095 | 0 | 0.798749 | 0.855215 | 0.657294 | -0.174313 |
| B1_query_ce_local | 0.142214 | 0.280407 | 0.256922 | 0.739805 | 0.87263 | 0 | 0.759562 | 0.694114 | 0.635514 | 0.0290819 |
| B2_query_hier_nolocal | 0.164758 | 0.18349 | 0.257905 | 0.819556 | 0.852369 | 0 | 0.734334 | 0.777665 | 0.5983 | -0.176327 |
| B3_query_hier_local | 0.15636 | 0.30853 | 0.344449 | 0.80649 | 0.864734 | 0 | 0.753872 | 0.804692 | 0.572881 | 0.0760461 |

## Interpretation Contract

- A1 > A0 suggests early high-resolution evidence matters.
- A2 > A0 suggests task query readout matters.
- A3 > A1/A2 suggests both local evidence and task readout are needed.
- B3 > B1/B2 suggests local supervision and coherent hierarchical probability are both useful.