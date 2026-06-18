# BUT Capacity Error Gap: cand6529_init_seed20261022_4ep_visual

Report-only diagnostic. This is not a formal model-selection result.

## Largest Test Buckets

| capacity_bucket | n | acc | good_n | good_recall | medium_n | medium_recall | bad_n | bad_recall | record_id | original_region | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| but_test | 1142 | 0.845884 | 646 | 0.899381 | 342 | 0.997076 | 154 | 0.285714 | nan | nan | nan |
| but_test | 278 | 0.888489 | 199 | 0.864322 | 75 | 1 | 4 | 0 | 113001 | nan | nan |
| but_test | 274 | 0.821168 | 70 | 0.742857 | 173 | 1 | 31 | 0 | 124001 | nan | nan |
| but_test | 231 | 0.675325 | 76 | 1 | 36 | 1 | 119 | 0.369748 | 122001 | nan | nan |
| but_test | 180 | 0.955556 | 167 | 0.958084 | 13 | 0.923077 | 0 | nan | 121001 | nan | nan |
| but_test | 179 | 0.927374 | 134 | 0.902985 | 45 | 1 | 0 | nan | 115001 | nan | nan |
| but_test | 162 | 0.95679 | 135 | 0.948148 | 27 | 1 | 0 | nan | 113001 | good_medium_overlap | nan |
| but_test | 135 | 0.948148 | 135 | 0.948148 | 0 | nan | 0 | nan | 113001 | good_medium_overlap | good |
| but_test | 127 | 0.968504 | 106 | 0.962264 | 21 | 1 | 0 | nan | 115001 | good_medium_overlap | nan |
| but_test | 122 | 0.95082 | 118 | 0.949153 | 4 | 1 | 0 | nan | 121001 | good_medium_overlap | nan |
| but_test | 121 | 0.595041 | 67 | 0.731343 | 23 | 1 | 31 | 0 | 124001 | outlier_low_confidence | nan |
| but_test | 119 | 0.369748 | 0 | nan | 0 | nan | 119 | 0.369748 | 122001 | near_bad_boundary | nan |
| but_test | 119 | 0.369748 | 0 | nan | 0 | nan | 119 | 0.369748 | 122001 | near_bad_boundary | bad |
| but_test | 118 | 0.949153 | 118 | 0.949153 | 0 | nan | 0 | nan | 121001 | good_medium_overlap | good |
| but_test | 106 | 0.962264 | 106 | 0.962264 | 0 | nan | 0 | nan | 115001 | good_medium_overlap | good |
| but_test | 93 | 1 | 3 | 1 | 90 | 1 | 0 | nan | 124001 | good_medium_overlap | nan |
| but_test | 90 | 1 | 0 | nan | 90 | 1 | 0 | nan | 124001 | good_medium_overlap | medium |
| but_test | 80 | 0.7 | 55 | 0.636364 | 21 | 1 | 4 | 0 | 113001 | outlier_low_confidence | nan |
| but_test | 77 | 1 | 55 | 1 | 22 | 1 | 0 | nan | 122001 | good_medium_overlap | nan |
| but_test | 67 | 0.731343 | 67 | 0.731343 | 0 | nan | 0 | nan | 124001 | outlier_low_confidence | good |
| but_test | 60 | 1 | 0 | nan | 60 | 1 | 0 | nan | 124001 | clean_core | medium |
| but_test | 60 | 1 | 0 | nan | 60 | 1 | 0 | nan | 124001 | clean_core | nan |
| but_test | 55 | 1 | 55 | 1 | 0 | nan | 0 | nan | 122001 | good_medium_overlap | good |
| but_test | 55 | 0.636364 | 55 | 0.636364 | 0 | nan | 0 | nan | 113001 | outlier_low_confidence | good |
| but_test | 44 | 0.954545 | 36 | 0.972222 | 8 | 0.875 | 0 | nan | 121001 | outlier_low_confidence | nan |

## Top Test Error Transitions

| capacity_bucket | record_id | original_region | true_label | pred_label | n |
| --- | --- | --- | --- | --- | --- |
| but_test | 113001 | good_medium_overlap | good | good | 128 |
| but_test | 121001 | good_medium_overlap | good | good | 112 |
| but_test | 115001 | good_medium_overlap | good | good | 102 |
| but_test | 124001 | good_medium_overlap | medium | medium | 90 |
| but_test | 122001 | near_bad_boundary | bad | medium | 75 |
| but_test | 124001 | clean_core | medium | medium | 60 |
| but_test | 122001 | good_medium_overlap | good | good | 55 |
| but_test | 124001 | outlier_low_confidence | good | good | 49 |
| but_test | 122001 | near_bad_boundary | bad | bad | 44 |
| but_test | 113001 | outlier_low_confidence | good | good | 35 |
| but_test | 121001 | outlier_low_confidence | good | good | 35 |
| but_test | 124001 | outlier_low_confidence | bad | medium | 31 |
| but_test | 113001 | clean_core | medium | medium | 27 |
| but_test | 113001 | good_medium_overlap | medium | medium | 27 |
| but_test | 115001 | clean_core | medium | medium | 24 |
| but_test | 124001 | outlier_low_confidence | medium | medium | 23 |
| but_test | 122001 | good_medium_overlap | medium | medium | 22 |
| but_test | 113001 | outlier_low_confidence | medium | medium | 21 |
| but_test | 115001 | good_medium_overlap | medium | medium | 21 |
| but_test | 122001 | clean_core | good | good | 21 |
| but_test | 113001 | outlier_low_confidence | good | medium | 20 |
| but_test | 115001 | clean_core | good | good | 18 |
| but_test | 124001 | outlier_low_confidence | good | medium | 18 |
| but_test | 121001 | clean_core | good | good | 13 |
| but_test | 113001 | clean_core | good | good | 9 |

## Top Correct-vs-Wrong Feature Gaps

| class_name | feature | ks_correct_vs_wrong | correct_median | wrong_median | delta_wrong_minus_correct | correct_p10 | correct_p90 | wrong_p10 | wrong_p90 | n_correct | n_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad | sqi_basSQI | 0.481818 | 0.985269 | 0.97499 | -0.0102785 | 0.973819 | 0.989997 | 0.838069 | 0.98579 | 44 | 110 |
| bad | lf_ratio | 0.477273 | 0.013508 | 0.0221209 | 0.00861292 | 0.00816922 | 0.0215733 | 0.0123789 | 0.142086 | 44 | 110 |
| bad | band_0p3_1 | 0.477273 | 0.013508 | 0.0221209 | 0.00861292 | 0.00816922 | 0.0215733 | 0.0123789 | 0.142086 | 44 | 110 |
| bad | pc1 | 0.468182 | 9.07989 | 8.72243 | -0.357451 | 8.75635 | 9.47897 | -0.406401 | 9.354 | 44 | 110 |
| bad | wavelet_e4 | 0.463636 | 0.18165 | 0.165123 | -0.0165265 | 0.167705 | 0.207999 | 0.00109773 | 0.196005 | 44 | 110 |
| bad | row_pos | 0.454545 | 32520.5 | 32568.5 | 48 | 32481.5 | 32568.6 | 32497.9 | 32640.2 | 44 | 110 |
| bad | pca_margin | 0.454545 | 5.46666 | 5.05766 | -0.408993 | 5.03845 | 5.87113 | -6.99625 | 5.64214 | 44 | 110 |
| bad | class_margin_percentile | 0.454545 | 0.0831599 | 0.0737938 | -0.00936613 | 0.0735289 | 0.0894607 | 0.0217219 | 0.0866793 | 44 | 110 |
| bad | pca_margin_rank | 0.454545 | 0.0831599 | 0.0737938 | -0.00936613 | 0.0735289 | 0.0894607 | 0.0217219 | 0.0866793 | 44 | 110 |
| bad | boundary_confidence | 0.454545 | 0.391616 | 0.387874 | -0.00374172 | 0.387734 | 0.39506 | 0.0740804 | 0.393393 | 44 | 110 |
| bad | region_confidence | 0.454545 | 0.305461 | 0.302542 | -0.00291854 | 0.302433 | 0.308147 | 0.0185201 | 0.306846 | 44 | 110 |
| bad | pc2 | 0.440909 | -1.05341 | -0.801283 | 0.252131 | -1.42796 | -0.805548 | -1.22785 | 6.19606 | 44 | 110 |
| bad | hjorth_complexity | 0.427273 | 1.24334 | 1.26228 | 0.0189406 | 1.22257 | 1.26 | 1.23202 | 2.80625 | 44 | 110 |
| bad | pca_nearest_other_distance | 0.427273 | 10.475 | 10.0127 | -0.462228 | 10.0566 | 11.1807 | 5.7779 | 10.8316 | 44 | 110 |
| bad | non_qrs_diff_p95 | 0.418182 | 0.418812 | 0.401059 | -0.0177528 | 0.397252 | 0.447965 | 0.109391 | 0.437644 | 44 | 110 |
| bad | hjorth_mobility | 0.413636 | 1.39258 | 1.37073 | -0.0218509 | 1.36783 | 1.42828 | 0.326686 | 1.41205 | 44 | 110 |
| bad | wavelet_e0 | 0.395455 | 0.227431 | 0.27093 | 0.0434992 | 0.183506 | 0.281108 | 0.211474 | 0.736937 | 44 | 110 |
| bad | zero_crossing_rate | 0.390909 | 0.493595 | 0.478783 | -0.0148118 | 0.478223 | 0.51193 | 0.0855885 | 0.509287 | 44 | 110 |
| bad | wavelet_e3 | 0.390909 | 0.170856 | 0.158572 | -0.0122838 | 0.153113 | 0.19002 | 0.0123805 | 0.186843 | 44 | 110 |
| bad | baseline_step | 0.381818 | 0.231594 | 0.29756 | 0.0659665 | 0.136423 | 0.309621 | 0.184701 | 0.786858 | 44 | 110 |
| bad | flatline_ratio | 0.377273 | 0.00800641 | 0.0104083 | 0.00240192 | 0.00640512 | 0.011209 | 0.00640512 | 0.0713371 | 44 | 110 |
| bad | sample_entropy_proxy | 0.377273 | 0.895225 | 0.884136 | -0.011089 | 0.877628 | 0.912919 | 0.684165 | 0.90395 | 44 | 110 |
| bad | band_1_5 | 0.363636 | 0.106051 | 0.122725 | 0.0166742 | 0.0871327 | 0.129615 | 0.0956406 | 0.653047 | 44 | 110 |
| bad | diff_abs_median | 0.354545 | 0.156516 | 0.150476 | -0.00603988 | 0.147267 | 0.163098 | 0.0205503 | 0.159224 | 44 | 110 |
| bad | spurious_peak_density | 0.354545 | 6.5 | 6.3 | -0.2 | 6.1 | 6.9 | 5.3 | 6.7 | 44 | 110 |
| bad | diff_abs_p95 | 0.35 | 0.439481 | 0.428635 | -0.010846 | 0.424368 | 0.449845 | 0.11231 | 0.45069 | 44 | 110 |
| bad | band_30_45 | 0.35 | 0.282116 | 0.257737 | -0.0243785 | 0.249348 | 0.328058 | 0.00511282 | 0.314931 | 44 | 110 |
| bad | higuchi_fd_proxy | 0.345455 | 1.98466 | 1.97291 | -0.0117547 | 1.96853 | 2.0021 | 1.26137 | 1.99685 | 44 | 110 |
| bad | non_qrs_rms_ratio | 0.345455 | 0.92238 | 0.903262 | -0.0191182 | 0.893398 | 0.980574 | 0.650279 | 0.96827 | 44 | 110 |
| bad | diff_zero_crossing_rate | 0.340909 | 0.675881 | 0.663061 | -0.0128205 | 0.657212 | 0.696635 | 0.229968 | 0.683574 | 44 | 110 |
| bad | local_rms_cv | 0.336364 | 0.0592298 | 0.0689972 | 0.00976742 | 0.0459669 | 0.080264 | 0.0474829 | 0.323595 | 44 | 110 |
| bad | detail_instability | 0.336364 | 0.280944 | 0.289036 | 0.00809128 | 0.267008 | 0.295249 | 0.274715 | 0.56156 | 44 | 110 |
| bad | hf_ratio | 0.336364 | 0.493308 | 0.471608 | -0.0216997 | 0.458581 | 0.529757 | 0.0257053 | 0.52054 | 44 | 110 |
| bad | qrs_slope_median | 0.336364 | 2.81621 | 2.71403 | -0.102171 | 2.63692 | 3.11956 | 0.773544 | 3.03624 | 44 | 110 |
| bad | sqi_kSQI | 0.336364 | 2.96709 | 3.06995 | 0.102858 | 2.72355 | 3.12296 | 2.851 | 5.86604 | 44 | 110 |

Summary CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_summary_by_bucket_record_region.csv`
Transitions CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_transitions.csv`
Feature gaps CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_test_error_feature_gaps.csv`
Test errors CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_test_errors.csv`
Waveform panel: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_capacity_error_gap\cand6529_init_seed20261022_4ep_visual_test_error_waveform_panels.png`
