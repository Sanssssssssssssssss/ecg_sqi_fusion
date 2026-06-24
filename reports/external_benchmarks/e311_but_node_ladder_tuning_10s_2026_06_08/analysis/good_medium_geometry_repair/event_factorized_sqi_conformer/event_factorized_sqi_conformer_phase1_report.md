# Event-Factorized SQI Conformer phase1 Report

- Generated: 2026-06-24 18:20:45
- Formal input contract: waveform-derived channels only.
- SQI/factor targets are training teacher/diagnostic targets only.
- All checkpoints in this stage are trained from scratch.

## Metrics

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_precision | medium_precision | bad_precision | good_to_medium | medium_to_good | bad_to_medium | confusion_3x3 | macro_f1_sklearn | supported_labels | bad_fpr_nonbad | artifact_positive_nonbad_count | artifact_positive_nonbad_bad_fpr | factor_mae | quality_subtype_rows | quality_subtype_acc | quality_subtype_class_acc | boundary_four_rows | boundary_label_acc | boundary_family_acc | boundary_label_balanced_acc | record_macro_acc | record_macro_supported_f1 | record_macro_full_f1 | bad_record_count | bad_containing_record_bad_recall_mean | bad_containing_record_acc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only_unified_subtype | clean_test | 24 | 0.333333 | 0.166667 | 1 | 0 | 0 | 0.333333 | 0 | 0 | 0 | 8 | 0 | [[8, 0, 0], [8, 0, 0], [8, 0, 0]] | 0.166667 | good,medium,bad | 0 | 4 | 0 | 0.579948 | 24 | 0.0833333 | 0.333333 | 1 | 0 | 1 | 0 | 0.335664 | 0.16732 | 0.16732 | 2 | 0 | 0.335664 |
| E1_query_only_unified_subtype | clean_test | 24 | 0.25 | 0.148148 | 0 | 0.75 | 0 | 0 | 0.315789 | 0 | 5 | 0 | 8 | [[0, 5, 3], [0, 6, 2], [0, 8, 0]] | 0.148148 | good,medium,bad | 0.3125 | 5 | 0.2 | 0.996371 | 24 | 0 | 0.333333 | 1 | 1 | 0 | 1 | 0.0555556 | 0.031746 | 0.031746 | 2 | 0 | 0.166667 |
| E1_query_only_unified_subtype | clean_val | 24 | 0.333333 | 0.166667 | 1 | 0 | 0 | 0.333333 | 0 | 0 | 0 | 8 | 0 | [[8, 0, 0], [8, 0, 0], [8, 0, 0]] | 0.166667 | good,medium,bad | 0 | 3 | 0 | 0.630029 | 24 | 0 | 0.333333 | 3 | 0.333333 | 0.666667 | 0.5 | 0.263889 | 0.156269 | 0.125966 | 2 | 0 | 0.208333 |
| E1_query_only_unified_subtype | clean_val | 24 | 0.166667 | 0.111111 | 0 | 0.5 | 0 | 0 | 0.25 | 0 | 4 | 0 | 8 | [[0, 4, 4], [0, 4, 4], [0, 8, 0]] | 0.111111 | good,medium,bad | 0.5 | 7 | 0.714286 | 0.824202 | 24 | 0.0416667 | 0.333333 | 5 | 0.4 | 0.6 | 0.5 | 0.166667 | 0.111111 | 0.111111 | 1 | 0 | 0.166667 |

## Feature Recovery

| candidate | bucket | feature | corr_all | mae | corr_good | corr_medium | corr_bad | corr_min_supported_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only_unified_subtype | clean_test | qrs_visibility | -0.137757 | 0.811817 | -0.788539 | -0.488036 | -0.649245 | -0.788539 |
| E1_query_only_unified_subtype | clean_test | detector_agreement | 0.536424 | 0.497478 | 0.229724 | 0.568912 | -0.470132 | -0.470132 |
| E1_query_only_unified_subtype | clean_test | baseline_step | 0.288037 | 1.55135 | 0.366658 | 0.0339742 | 0.369412 | 0.0339742 |
| E1_query_only_unified_subtype | clean_test | flatline_ratio | -0.461568 | 0.478077 | -0.821128 | 0.494138 | 0.386557 | -0.821128 |
| E1_query_only_unified_subtype | clean_test | sqi_basSQI | -0.217812 | 1.41818 | -0.347022 | -0.205645 | 0.360266 | -0.347022 |
| E1_query_only_unified_subtype | clean_test | non_qrs_diff_p95 | 0.671549 | 2.83225 | 0.46985 | 0.459659 | 0.170329 | 0.170329 |
| E1_query_only_unified_subtype | clean_test | non_qrs_rms_ratio | -0.245294 | 0.607812 | 0.418644 | -0.212257 | -0.466859 | -0.466859 |
| E1_query_only_unified_subtype | clean_test | qrs_band_ratio | -0.23176 | 1.00652 | -0.427821 | 0.674986 | 0.0562466 | -0.427821 |
| E1_query_only_unified_subtype | clean_test | template_corr | 0.0722547 | 0.57118 | 0.512499 | 0.348724 | 0.247914 | 0.247914 |
| E1_query_only_unified_subtype | clean_test | amplitude_entropy | -0.658237 | 0.776561 | -0.656801 | -0.717765 | 0.2133 | -0.717765 |
| E1_query_only_unified_subtype | clean_test | contact_loss_win_ratio | 0 | 0.408861 | nan | nan | nan | nan |
| E1_query_only_unified_subtype | clean_test | qrs_visibility | -0.366584 | 0.216694 | -0.254879 | 0.111526 | 0.147184 | -0.254879 |
| E1_query_only_unified_subtype | clean_test | detector_agreement | 0.24142 | 0.564636 | 0.382244 | -0.0482123 | 0.149558 | -0.0482123 |
| E1_query_only_unified_subtype | clean_test | baseline_step | 0.402306 | 1.14456 | 0.851625 | 0.160889 | 0.00777288 | 0.00777288 |
| E1_query_only_unified_subtype | clean_test | flatline_ratio | 0.480627 | 1.07669 | 0.360027 | -0.322695 | 0.679931 | -0.322695 |
| E1_query_only_unified_subtype | clean_test | sqi_basSQI | -0.0047807 | 0.415971 | -0.705727 | 0.182186 | 0.190153 | -0.705727 |
| E1_query_only_unified_subtype | clean_test | non_qrs_diff_p95 | 0.140664 | 0.6298 | -0.332412 | -0.639154 | 0.604367 | -0.639154 |
| E1_query_only_unified_subtype | clean_test | non_qrs_rms_ratio | 0.479089 | 0.257071 | 0.798406 | -0.557717 | 0.177797 | -0.557717 |
| E1_query_only_unified_subtype | clean_test | qrs_band_ratio | -0.29921 | 0.473158 | -0.448308 | -0.355649 | -0.0979162 | -0.448308 |
| E1_query_only_unified_subtype | clean_test | template_corr | 0.406341 | 0.769955 | -0.0300757 | 0.173702 | 0.31907 | -0.0300757 |
| E1_query_only_unified_subtype | clean_test | amplitude_entropy | -0.0169251 | 0.304593 | 0.341428 | -0.114988 | 0.499789 | -0.114988 |
| E1_query_only_unified_subtype | clean_test | contact_loss_win_ratio | 0.613476 | 0.526299 | nan | nan | 0.535299 | 0.535299 |

## Record Metrics Preview

| candidate | record_id | rows | good_rows | medium_rows | bad_rows | full_macro_f1 | supported_macro_f1 | acc | good_recall | medium_recall | bad_recall | artifact_positive_nonbad | artifact_positive_nonbad_bad_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_query_only_unified_subtype | 100001 | 24 | 8 | 8 | 8 | 0.111111 | 0.111111 | 0.166667 | 0 | 0.5 | 0 | 7 | 0.714286 |
| E1_query_only_unified_subtype | 100002 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| E1_query_only_unified_subtype | 104001 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| E1_query_only_unified_subtype | 105001 | 18 | 5 | 7 | 6 | 0.190476 | 0.190476 | 0.333333 | 0 | 0.857143 | 0 | 2 | 0 |
| E1_query_only_unified_subtype | 114001 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan |
| E1_query_only_unified_subtype | 122001 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan |
| E1_query_only_unified_subtype | 123001 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| E1_query_only_unified_subtype | 104001 | 8 | 3 | 5 | 0 | 0.181818 | 0.272727 | 0.375 | 1 | 0 | 0 | 3 | 0 |
| E1_query_only_unified_subtype | 114001 | 12 | 5 | 2 | 5 | 0.196078 | 0.196078 | 0.416667 | 1 | 0 | 0 | 0 | nan |
| E1_query_only_unified_subtype | 122001 | 4 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan |
| E1_query_only_unified_subtype | 100001 | 11 | 4 | 6 | 1 | 0.177778 | 0.177778 | 0.363636 | 1 | 0 | 0 | 3 | 0 |
| E1_query_only_unified_subtype | 111001 | 13 | 4 | 2 | 7 | 0.156863 | 0.156863 | 0.307692 | 1 | 0 | 0 | 1 | 0 |

## Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_metrics.csv`
- Feature recovery: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_feature_recovery.csv`
- Record metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\phase1_record_metrics.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\event_factorized_sqi_conformer`