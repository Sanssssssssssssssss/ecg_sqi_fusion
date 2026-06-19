# BUT Capacity Error Gap: pba_cvfold2_wavefactgm_v3_best

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 4316 | 0.934893 | 2244 | 0.935829 | 1255 | 0.890837 | 817 | 1 | nan | nan | nan |
| but_test | 2154 | 0.955432 | 910 | 0.981319 | 451 | 0.824834 | 793 | 1 | 105001 | nan | nan |
| but_test | 1225 | 0.907755 | 891 | 0.900112 | 334 | 0.928144 | 0 | nan | 100001 | nan | nan |
| but_test | 942 | 0.88535 | 683 | 0.877013 | 259 | 0.907336 | 0 | nan | 100001 | good_medium_overlap | nan |
| but_test | 830 | 0.895181 | 528 | 0.967803 | 302 | 0.768212 | 0 | nan | 105001 | good_medium_overlap | nan |
| but_test | 793 | 1 | 0 | nan | 0 | nan | 793 | 1 | 105001 | right_bad_island | nan |
| but_test | 793 | 1 | 0 | nan | 0 | nan | 793 | 1 | 105001 | right_bad_island | bad |
| but_test | 683 | 0.877013 | 683 | 0.877013 | 0 | nan | 0 | nan | 100001 | good_medium_overlap | good |
| but_test | 596 | 0.926174 | 185 | 0.940541 | 411 | 0.919708 | 0 | nan | 111001 | nan | nan |
| but_test | 530 | 0.983019 | 382 | 1 | 148 | 0.939189 | 0 | nan | 105001 | clean_core | nan |
| but_test | 528 | 0.967803 | 528 | 0.967803 | 0 | nan | 0 | nan | 105001 | good_medium_overlap | good |
| but_test | 498 | 0.917671 | 183 | 0.939891 | 315 | 0.904762 | 0 | nan | 111001 | good_medium_overlap | nan |
| but_test | 382 | 1 | 382 | 1 | 0 | nan | 0 | nan | 105001 | clean_core | good |
| but_test | 315 | 0.904762 | 0 | nan | 315 | 0.904762 | 0 | nan | 111001 | good_medium_overlap | medium |
| but_test | 302 | 0.768212 | 0 | nan | 302 | 0.768212 | 0 | nan | 105001 | good_medium_overlap | medium |
| but_test | 283 | 0.982332 | 208 | 0.975962 | 75 | 1 | 0 | nan | 100001 | clean_core | nan |
| but_test | 259 | 0.907336 | 0 | nan | 259 | 0.907336 | 0 | nan | 100001 | good_medium_overlap | medium |
| but_test | 208 | 0.975962 | 208 | 0.975962 | 0 | nan | 0 | nan | 100001 | clean_core | good |
| but_test | 183 | 0.939891 | 183 | 0.939891 | 0 | nan | 0 | nan | 111001 | good_medium_overlap | good |
| but_test | 148 | 0.939189 | 0 | nan | 148 | 0.939189 | 0 | nan | 105001 | clean_core | medium |
| but_test | 97 | 0.969072 | 2 | 1 | 95 | 0.968421 | 0 | nan | 111001 | clean_core | nan |
| but_test | 95 | 0.968421 | 0 | nan | 95 | 0.968421 | 0 | nan | 111001 | clean_core | medium |
| but_test | 75 | 1 | 0 | nan | 75 | 1 | 0 | nan | 100001 | clean_core | medium |
| but_test | 48 | 0.895833 | 42 | 0.880952 | 6 | 1 | 0 | nan | 114001 | nan | nan |
| but_test | 46 | 0.891304 | 42 | 0.880952 | 4 | 1 | 0 | nan | 114001 | good_medium_overlap | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 105001 | right_bad_island | bad | bad | 793 |
| but_test | 100001 | good_medium_overlap | good | good | 599 |
| but_test | 105001 | good_medium_overlap | good | good | 511 |
| but_test | 105001 | clean_core | good | good | 382 |
| but_test | 111001 | good_medium_overlap | medium | medium | 285 |
| but_test | 100001 | good_medium_overlap | medium | medium | 235 |
| but_test | 105001 | good_medium_overlap | medium | medium | 232 |
| but_test | 100001 | clean_core | good | good | 203 |
| but_test | 111001 | good_medium_overlap | good | good | 172 |
| but_test | 105001 | clean_core | medium | medium | 139 |
| but_test | 111001 | clean_core | medium | medium | 92 |
| but_test | 100001 | good_medium_overlap | good | medium | 84 |
| but_test | 100001 | clean_core | medium | medium | 75 |
| but_test | 105001 | good_medium_overlap | medium | good | 70 |
| but_test | 114001 | good_medium_overlap | good | good | 37 |
| but_test | 111001 | good_medium_overlap | medium | good | 29 |
| but_test | 100001 | good_medium_overlap | medium | good | 24 |
| but_test | 122001 | near_bad_boundary | bad | bad | 24 |
| but_test | 118001 | good_medium_overlap | good | good | 23 |
| but_test | 115001 | good_medium_overlap | good | good | 20 |
| but_test | 103002 | clean_core | good | good | 19 |
| but_test | 103001 | good_medium_overlap | good | good | 17 |
| but_test | 105001 | good_medium_overlap | good | medium | 17 |
| but_test | 113001 | good_medium_overlap | good | good | 16 |
| but_test | 121001 | good_medium_overlap | good | good | 16 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| good | sqi_sSQI | 0.630079 | 4.19004 | 3.39638 | -0.793657 | 3.5091 | 4.99254 | 1.24317 | 3.99603 | 2100 | 144 |
| good | amplitude_entropy | 0.619087 | 0.609266 | 0.69234 | 0.0830736 | 0.54131 | 0.682341 | 0.635581 | 0.76585 | 2100 | 144 |
| good | sqi_kSQI | 0.588135 | 23.5758 | 18.1025 | -5.47323 | 18.3719 | 32.495 | 9.85542 | 22.0943 | 2100 | 144 |
| good | low_amp_ratio | 0.549603 | 0.3552 | 0.2272 | -0.128 | 0.2056 | 0.4784 | 0.14288 | 0.3132 | 2100 | 144 |
| good | baseline_step | 0.53119 | 0.261318 | 0.572094 | 0.310775 | 0.133345 | 0.632362 | 0.2941 | 1.02428 | 2100 | 144 |
| good | qrs_prom_p90 | 0.496944 | 6.39486 | 5.95354 | -0.441325 | 5.91375 | 7.58405 | 5.02487 | 6.41436 | 2100 | 144 |
| good | pca_margin | 0.470159 | 2.50291 | 0.477573 | -2.02534 | 0.187572 | 3.85238 | -0.89794 | 2.7532 | 2100 | 144 |
| good | class_margin_percentile | 0.470159 | 0.657308 | 0.255677 | -0.401631 | 0.20629 | 0.93835 | 0.0851611 | 0.715426 | 2100 | 144 |
| good | pca_margin_rank | 0.470159 | 0.657308 | 0.255677 | -0.401631 | 0.20629 | 0.93835 | 0.0851611 | 0.715426 | 2100 | 144 |
| good | pc1 | 0.433056 | -4.2771 | -2.52404 | 1.75306 | -5.97714 | -2.22891 | -4.6323 | -1.20661 | 2100 | 144 |
| good | boundary_confidence | 0.4325 | 0.749617 | 0.552249 | -0.197368 | 0.517271 | 1.07253 | 0.384842 | 0.790338 | 2100 | 144 |
| good | region_confidence | 0.4325 | 0.659663 | 0.485979 | -0.173684 | 0.455198 | 1.07253 | 0.338661 | 0.695498 | 2100 | 144 |
| good | rms | 0.422659 | 0.309927 | 0.245871 | -0.0640559 | 0.242192 | 0.323399 | 0.194543 | 0.323399 | 2100 | 144 |
| good | qrs_prom_median | 0.416667 | 5.56553 | 4.97063 | -0.594898 | 1.56296 | 6.57834 | 2.14288 | 5.55426 | 2100 | 144 |
| good | knn_label_purity | 0.411865 | 1 | 0.9 | -0.1 | 0.866667 | 1 | 0.633333 | 1 | 2100 | 144 |
| good | pc6 | 0.394603 | -0.0297026 | -0.64378 | -0.614078 | -1.40825 | 1.49587 | -2.06997 | 0.0205846 | 2100 | 144 |
| good | std | 0.387778 | 0.29715 | 0.240855 | -0.0562945 | 0.233717 | 0.316172 | 0.192877 | 0.31603 | 2100 | 144 |
| good | hjorth_activity | 0.387778 | 0.088298 | 0.058012 | -0.030286 | 0.0546238 | 0.0999649 | 0.0372027 | 0.0998747 | 2100 | 144 |
| good | non_qrs_rms_ratio | 0.383373 | 0.307846 | 0.385037 | 0.0771911 | 0.122089 | 0.463529 | 0.292737 | 0.573895 | 2100 | 144 |
| good | pca_nearest_other_distance | 0.372341 | 6.8283 | 4.69373 | -2.13458 | 4.3454 | 9.28503 | 3.37955 | 8.05169 | 2100 | 144 |
| good | sample_entropy_proxy | 0.343968 | 0.314381 | 0.358752 | 0.0443708 | 0.233569 | 0.373748 | 0.290159 | 0.428232 | 2100 | 144 |
| good | band_1_5 | 0.338333 | 0.275121 | 0.210703 | -0.0644178 | 0.19024 | 0.350021 | 0.165483 | 0.32299 | 2100 | 144 |
| good | pc4 | 0.329127 | 0.172043 | -0.559205 | -0.731248 | -1.1236 | 4.58877 | -1.72977 | 1.39254 | 2100 | 144 |
| good | ptp_p99_p01 | 0.323175 | 1.90162 | 1.6285 | -0.273124 | 1.42701 | 2.10435 | 1.13426 | 1.98279 | 2100 | 144 |
| good | lf_ratio | 0.321429 | 0.0149883 | 0.034054 | 0.0190656 | 0.0028285 | 0.172111 | 0.00861853 | 0.141735 | 2100 | 144 |
| good | band_0p3_1 | 0.321429 | 0.0149883 | 0.034054 | 0.0190656 | 0.0028285 | 0.172111 | 0.00861853 | 0.141735 | 2100 | 144 |
| good | pc7 | 0.294405 | -0.0953248 | 0.517896 | 0.61322 | -1.27434 | 1.02994 | -0.539959 | 2.05479 | 2100 | 144 |
| good | flatline_ratio | 0.292738 | 0.298239 | 0.194155 | -0.104083 | 0.151321 | 0.448359 | 0.122178 | 0.358687 | 2100 | 144 |
| good | diff_abs_p95 | 0.275556 | 0.17351 | 0.222369 | 0.0488593 | 0.0806408 | 0.318291 | 0.0848885 | 0.430873 | 2100 | 144 |
| good | source_idx | 0.262976 | 17690 | 6900 | -10790 | 1332.2 | 29850.1 | 1840.5 | 31027.3 | 2100 | 144 |
| good | sqi_bSQI | 0.251349 | 0.9 | 0.916667 | 0.0166667 | 0.818182 | 0.923077 | 0.833333 | 0.9375 | 2100 | 144 |
| good | qrs_visibility | 0.245754 | 0.585099 | 0.507837 | -0.0772614 | 0.214052 | 0.831338 | 0.297648 | 0.688963 | 2100 | 144 |
| good | qrs_band_ratio | 0.243373 | 0.513024 | 0.548582 | 0.0355585 | 0.430855 | 0.60842 | 0.442014 | 0.628747 | 2100 | 144 |
| good | medium_detail_unreliable_score | 0.239365 | 0.580142 | 0.507318 | -0.0728239 | 0.190876 | 0.829994 | 0.281231 | 0.688963 | 2100 | 144 |
| good | rr_count_detector_c | 0.231548 | 15 | 19 | 4 | 11 | 23 | 12 | 24.7 | 2100 | 144 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_wavefactgm_v3_best_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_wavefactgm_v3_best_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_wavefactgm_v3_best_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_wavefactgm_v3_best_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\pba_cvfold2_wavefactgm_v3_best_test_error_waveform_panels.png`
