# BUT Split Distribution Shift Audit

- Generated: 2026-06-21 03:00:27
- Purpose: check whether BUT test classes are inside the train distribution used as PTB synthetic target.

## Split Counts

| split | class_name | n |
| --- | --- | --- |
| test | bad | 118 |
| test | good | 1004 |
| test | medium | 2077 |
| train | bad | 3963 |
| train | good | 9603 |
| train | medium | 4145 |
| val | bad | 1 |
| val | good | 621 |
| val | medium | 43 |

## Largest Test-vs-Train Feature Shifts

### good

| class_name | feature | train_n | test_n | train_median | test_median | robust_gap | abs_gap | ks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good | non_qrs_diff_p95 | 9603 | 1004 | 1.76111 | 0.680265 | -0.894654 | 0.894654 | 0.800842 |
| good | band_15_30 | 9603 | 1004 | 0.223687 | 0.179661 | -0.86754 | 0.86754 | 0.45376 |
| good | detector_agreement | 9603 | 1004 | 0.416667 | 0.833333 | 0.833333 | 0.833333 | 0.581526 |
| good | band_30_45 | 9603 | 1004 | 0.0167997 | 0.0231509 | 0.649655 | 0.649655 | 0.388817 |
| good | mean_abs | 9603 | 1004 | 1.33926 | 1.2151 | -0.54518 | 0.54518 | 0.315653 |
| good | flatline_ratio | 9603 | 1004 | 0.11209 | 0.147318 | 0.473118 | 0.473118 | 0.28007 |
| good | rms | 9603 | 1004 | 3.03166 | 2.67798 | -0.417606 | 0.417606 | 0.227105 |
| good | amplitude_entropy | 9603 | 1004 | 0.758615 | 0.773808 | 0.223214 | 0.223214 | 0.204352 |
| good | template_corr | 9603 | 1004 | 0.883684 | 0.874891 | -0.20871 | 0.20871 | 0.312767 |
| good | baseline_step | 9603 | 1004 | 0.29666 | 0.319683 | 0.20666 | 0.20666 | 0.142485 |
| good | sqi_basSQI | 9603 | 1004 | 0.457323 | 0.438841 | -0.1982 | 0.1982 | 0.142485 |
| good | low_amp_ratio | 9603 | 1004 | 0.2424 | 0.2328 | -0.166667 | 0.166667 | 0.10811 |

### medium

| class_name | feature | train_n | test_n | train_median | test_median | robust_gap | abs_gap | ks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | band_30_45 | 4145 | 2077 | 0.0227789 | 0.0381995 | 0.964812 | 0.964812 | 0.400894 |
| medium | mean_abs | 4145 | 2077 | 1.11185 | 1.01298 | -0.432033 | 0.432033 | 0.291863 |
| medium | band_15_30 | 4145 | 2077 | 0.236432 | 0.204857 | -0.416419 | 0.416419 | 0.20353 |
| medium | template_corr | 4145 | 2077 | 0.901203 | 0.849193 | -0.407162 | 0.407162 | 0.219249 |
| medium | amplitude_entropy | 4145 | 2077 | 0.8068 | 0.795204 | -0.362694 | 0.362694 | 0.209242 |
| medium | non_qrs_diff_p95 | 4145 | 2077 | 2.49678 | 2.05425 | -0.353432 | 0.353432 | 0.216794 |
| medium | detector_agreement | 4145 | 2077 | 0.583333 | 0.5 | -0.333333 | 0.333333 | 0.170639 |
| medium | rms | 4145 | 2077 | 1.9779 | 1.76039 | -0.249047 | 0.249047 | 0.230322 |
| medium | low_amp_ratio | 4145 | 2077 | 0.2112 | 0.2048 | -0.205128 | 0.205128 | 0.144447 |
| medium | sqi_basSQI | 4145 | 2077 | 0.595406 | 0.61689 | 0.144531 | 0.144531 | 0.150028 |
| medium | baseline_step | 4145 | 2077 | 0.169881 | 0.155258 | -0.141514 | 0.141514 | 0.150028 |
| medium | qrs_band_ratio | 4145 | 2077 | 8.13879 | 8.15901 | 0.0052971 | 0.0052971 | 0.111797 |

### bad

| class_name | feature | train_n | test_n | train_median | test_median | robust_gap | abs_gap | ks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | band_30_45 | 3963 | 118 | 0.0567785 | 0.191158 | 18.0204 | 18.0204 | 1 |
| bad | band_15_30 | 3963 | 118 | 0.457773 | 0.22311 | -13.8027 | 13.8027 | 1 |
| bad | baseline_step | 3963 | 118 | 0.0426127 | 0.0568631 | 4.58175 | 4.58175 | 0.989759 |
| bad | sqi_basSQI | 3963 | 118 | 0.854372 | 0.814695 | -4.37047 | 4.37047 | 0.989759 |
| bad | mean_abs | 3963 | 118 | 0.858277 | 0.79641 | -2.38112 | 2.38112 | 0.900642 |
| bad | rms | 3963 | 118 | 1.0984 | 0.997684 | -2.36778 | 2.36778 | 0.913216 |
| bad | amplitude_entropy | 3963 | 118 | 0.826685 | 0.806019 | -2.29157 | 2.29157 | 0.844896 |
| bad | qrs_visibility | 3963 | 118 | 0.954959 | 1.06059 | 1.79092 | 1.79092 | 0.664054 |
| bad | flatline_ratio | 3963 | 118 | 0.00560448 | 0.00800641 | 1 | 1 | 0.469523 |
| bad | non_qrs_diff_p95 | 3963 | 118 | 2.77741 | 2.69527 | -0.724122 | 0.724122 | 0.386625 |
| bad | low_amp_ratio | 3963 | 118 | 0.2056 | 0.1992 | -0.499999 | 0.499999 | 0.261735 |
| bad | qrs_band_ratio | 3963 | 118 | 3.08993 | 3.16416 | 0.366118 | 0.366118 | 0.261448 |

## Figures

- good shift bar: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_test_train_good_feature_shift.png`
- good boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_split_good_feature_boxplots.png`
- medium shift bar: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_test_train_medium_feature_shift.png`
- medium boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_split_medium_feature_boxplots.png`
- bad shift bar: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_test_train_bad_feature_shift.png`
- bad boxplots: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_split_distribution_shift_audit\but_split_bad_feature_boxplots.png`