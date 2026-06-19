# BUT Capacity Error Gap: pba_cvfold2_cleantrainbad_b035_seed1840_failure

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 4316 | 0.691613 | 2244 | 0.98975 | 1255 | 0.608765 | 817 | 0 | nan | nan | nan |
| but_test | 2154 | 0.528784 | 910 | 0.994505 | 451 | 0.518847 | 793 | 0 | 105001 | nan | nan |
| but_test | 1225 | 0.887347 | 891 | 0.989899 | 334 | 0.613772 | 0 | nan | 100001 | nan | nan |
| but_test | 942 | 0.859873 | 683 | 0.986823 | 259 | 0.525097 | 0 | nan | 100001 | good_medium_overlap | nan |
| but_test | 830 | 0.779518 | 528 | 0.99053 | 302 | 0.410596 | 0 | nan | 105001 | good_medium_overlap | nan |
| but_test | 793 | 0 | 0 | nan | 0 | nan | 793 | 0 | 105001 | right_bad_island | nan |
| but_test | 793 | 0 | 0 | nan | 0 | nan | 793 | 0 | 105001 | right_bad_island | bad |
| but_test | 683 | 0.986823 | 683 | 0.986823 | 0 | nan | 0 | nan | 100001 | good_medium_overlap | good |
| but_test | 596 | 0.77349 | 185 | 0.994595 | 411 | 0.673966 | 0 | nan | 111001 | nan | nan |
| but_test | 530 | 0.926415 | 382 | 1 | 148 | 0.736486 | 0 | nan | 105001 | clean_core | nan |
| but_test | 528 | 0.99053 | 528 | 0.99053 | 0 | nan | 0 | nan | 105001 | good_medium_overlap | good |
| but_test | 498 | 0.753012 | 183 | 0.994536 | 315 | 0.612698 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 382 | 1 | 382 | 1 | 0 | nan | 0 | nan | 105001 | clean_core | good |
| but_test | 315 | 0.612698 | 0 | nan | 315 | 0.612698 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 302 | 0.410596 | 0 | nan | 302 | 0.410596 | 0 | nan | 105001 | good_medium_overlap | medium |
| but_test | 283 | 0.978799 | 208 | 1 | 75 | 0.92 | 0 | nan | 100001 | clean_core | nan |
| but_test | 259 | 0.525097 | 0 | nan | 259 | 0.525097 | 0 | nan | 100001 | good_medium_overlap | medium |
| but_test | 208 | 1 | 208 | 1 | 0 | nan | 0 | nan | 100001 | clean_core | good |
| but_test | 183 | 0.994536 | 183 | 0.994536 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 148 | 0.736486 | 0 | nan | 148 | 0.736486 | 0 | nan | 105001 | clean_core | medium |
| but_test | 97 | 0.876289 | 2 | 1 | 95 | 0.873684 | 0 | nan | 111001 | clean_core | nan |
| but_test | 95 | 0.873684 | 0 | nan | 95 | 0.873684 | 0 | nan | 111001 | clean_core | medium |
| but_test | 75 | 0.92 | 0 | nan | 75 | 0.92 | 0 | nan | 100001 | clean_core | medium |
| but_test | 48 | 0.9375 | 42 | 0.97619 | 6 | 0.666667 | 0 | nan | 114001 | nan | nan |
| but_test | 46 | 0.956522 | 42 | 0.97619 | 4 | 0.75 | 0 | nan | 114001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 105001 | right_bad_island | bad | medium | 793 |
| but_test | 100001 | good_medium_overlap | good | good | 674 |
| but_test | 105001 | good_medium_overlap | good | good | 523 |
| but_test | 105001 | clean_core | good | good | 382 |
| but_test | 100001 | clean_core | good | good | 208 |
| but_test | 111001 | good_medium_overlap | medium | medium | 193 |
| but_test | 111001 | good_medium_overlap | good | good | 182 |
| but_test | 105001 | good_medium_overlap | medium | good | 178 |
| but_test | 100001 | good_medium_overlap | medium | medium | 136 |
| but_test | 105001 | good_medium_overlap | medium | medium | 124 |
| but_test | 100001 | good_medium_overlap | medium | good | 123 |
| but_test | 111001 | good_medium_overlap | medium | good | 122 |
| but_test | 105001 | clean_core | medium | medium | 109 |
| but_test | 111001 | clean_core | medium | medium | 83 |
| but_test | 100001 | clean_core | medium | medium | 69 |
| but_test | 114001 | good_medium_overlap | good | good | 41 |
| but_test | 105001 | clean_core | medium | good | 39 |
| but_test | 122001 | near_bad_boundary | bad | medium | 24 |
| but_test | 118001 | good_medium_overlap | good | good | 23 |
| but_test | 126001 | good_medium_overlap | good | good | 21 |
| but_test | 113001 | good_medium_overlap | good | good | 20 |
| but_test | 115001 | good_medium_overlap | good | good | 20 |
| but_test | 103002 | clean_core | good | good | 19 |
| but_test | 103001 | good_medium_overlap | good | good | 17 |
| but_test | 121001 | good_medium_overlap | good | good | 16 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good | sqi_sSQI | 0.911301 | 4.16431 | 1.96205 | -2.20226 | 3.4254 | 4.97751 | 1.07167 | 3.10926 | 2221 | 23 |
| good | sqi_kSQI | 0.879079 | 23.3328 | 10.995 | -12.3378 | 17.8631 | 32.2212 | 6.99264 | 15.5967 | 2221 | 23 |
| good | ptp_p99_p01 | 0.870779 | 1.89089 | 1.18388 | -0.707014 | 1.41387 | 2.10068 | 1.02824 | 1.4336 | 2221 | 23 |
| good | rms | 0.865376 | 0.30805 | 0.195302 | -0.112748 | 0.238989 | 0.323399 | 0.183339 | 0.235019 | 2221 | 23 |
| good | std | 0.843764 | 0.295614 | 0.194506 | -0.101108 | 0.230305 | 0.316179 | 0.183014 | 0.233231 | 2221 | 23 |
| good | hjorth_activity | 0.843764 | 0.0873876 | 0.0378326 | -0.049555 | 0.0530404 | 0.0999693 | 0.0334942 | 0.0543969 | 2221 | 23 |
| good | amplitude_entropy | 0.824149 | 0.612664 | 0.753152 | 0.140488 | 0.542703 | 0.69151 | 0.689454 | 0.804986 | 2221 | 23 |
| good | qrs_prom_p90 | 0.822093 | 6.36861 | 5.45474 | -0.913868 | 5.87808 | 7.55569 | 4.54099 | 5.85553 | 2221 | 23 |
| good | hjorth_complexity | 0.674099 | 1.69738 | 2.03564 | 0.338255 | 1.59068 | 1.89041 | 1.62485 | 2.36919 | 2221 | 23 |
| good | low_amp_ratio | 0.663352 | 0.3488 | 0.2024 | -0.1464 | 0.1992 | 0.476 | 0.12304 | 0.29296 | 2221 | 23 |
| good | hjorth_mobility | 0.661942 | 0.610884 | 0.498178 | -0.112707 | 0.550875 | 0.667395 | 0.425306 | 0.681957 | 2221 | 23 |
| good | baseline_step | 0.63389 | 0.272901 | 0.824294 | 0.551393 | 0.135624 | 0.656536 | 0.420419 | 1.38991 | 2221 | 23 |
| good | non_qrs_rms_ratio | 0.630621 | 0.311485 | 0.538966 | 0.22748 | 0.128962 | 0.465852 | 0.327642 | 0.60368 | 2221 | 23 |
| good | wavelet_e0 | 0.615312 | 0.440246 | 0.668247 | 0.228001 | 0.316647 | 0.607338 | 0.349977 | 0.728369 | 2221 | 23 |
| good | wavelet_e1 | 0.607913 | 0.291231 | 0.179367 | -0.111864 | 0.195887 | 0.374503 | 0.138297 | 0.318363 | 2221 | 23 |
| good | lf_ratio | 0.591625 | 0.0158884 | 0.0525826 | 0.0366942 | 0.00288167 | 0.171596 | 0.026408 | 0.16018 | 2221 | 23 |
| good | band_0p3_1 | 0.591625 | 0.0158884 | 0.0525826 | 0.0366942 | 0.00288167 | 0.171596 | 0.026408 | 0.16018 | 2221 | 23 |
| good | pca_margin | 0.585263 | 2.43191 | 0.87923 | -1.55268 | 0.0732989 | 3.83664 | -0.875104 | 1.7777 | 2221 | 23 |
| good | class_margin_percentile | 0.585263 | 0.638209 | 0.317902 | -0.320307 | 0.191457 | 0.935164 | 0.0875433 | 0.48096 | 2221 | 23 |
| good | pca_margin_rank | 0.585263 | 0.638209 | 0.317902 | -0.320307 | 0.191457 | 0.935164 | 0.0875433 | 0.48096 | 2221 | 23 |
| good | pc7 | 0.577296 | -0.0600494 | 1.00422 | 1.06427 | -1.24309 | 1.07363 | 0.256799 | 2.43604 | 2221 | 23 |
| good | qrs_prom_median | 0.554451 | 5.51251 | 4.03107 | -1.48145 | 1.5716 | 6.55503 | 2.20835 | 5.11414 | 2221 | 23 |
| good | boundary_confidence | 0.552004 | 0.741421 | 0.539991 | -0.20143 | 0.499851 | 1.06758 | 0.350919 | 0.704588 | 2221 | 23 |
| good | region_confidence | 0.552004 | 0.65245 | 0.475192 | -0.177258 | 0.439869 | 1.06758 | 0.308809 | 0.620038 | 2221 | 23 |
| good | pc6 | 0.546483 | -0.0678644 | -1.14075 | -1.07288 | -1.42366 | 1.45082 | -2.10626 | 0.585683 | 2221 | 23 |
| good | pc2 | 0.537282 | -1.45438 | 2.66775 | 4.12213 | -2.88297 | 2.7666 | -1.49593 | 4.8961 | 2221 | 23 |
| good | wavelet_e3 | 0.510855 | 0.0654921 | 0.0417959 | -0.0236962 | 0.0416078 | 0.0931757 | 0.0251956 | 0.087969 | 2221 | 23 |
| good | non_qrs_diff_p95 | 0.506548 | 0.0549206 | 0.0335679 | -0.0213527 | 0.0240732 | 0.078943 | 0.0247993 | 0.0565911 | 2221 | 23 |
| good | local_rms_cv | 0.477732 | 0.130576 | 0.191725 | 0.0611489 | 0.0425327 | 0.2394 | 0.10872 | 0.290745 | 2221 | 23 |
| good | wavelet_e2 | 0.477419 | 0.184936 | 0.117543 | -0.0673922 | 0.123671 | 0.250631 | 0.0809456 | 0.246553 | 2221 | 23 |
| good | pc4 | 0.453478 | 0.130042 | -0.802466 | -0.932508 | -1.1409 | 4.53691 | -3.08006 | 0.714651 | 2221 | 23 |
| good | pc1 | 0.452499 | -4.20908 | -2.51095 | 1.69813 | -5.90889 | -2.11183 | -4.20852 | -1.12005 | 2221 | 23 |
| good | qrs_visibility | 0.449484 | 0.580807 | 0.455571 | -0.125236 | 0.218163 | 0.82611 | 0.261267 | 0.577271 | 2221 | 23 |
| good | wavelet_e4 | 0.444101 | 0.00560632 | 0.0034143 | -0.00219202 | 0.00326465 | 0.00865668 | 0.0021838 | 0.00935242 | 2221 | 23 |
| good | medium_detail_unreliable_score | 0.438678 | 0.575131 | 0.455571 | -0.11956 | 0.199839 | 0.825345 | 0.252545 | 0.577271 | 2221 | 23 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleantrainbad_b035_seed1840_failure_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleantrainbad_b035_seed1840_failure_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleantrainbad_b035_seed1840_failure_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleantrainbad_b035_seed1840_failure_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_cleantrainbad_b035_seed1840_failure_test_error_waveform_panels.png`
